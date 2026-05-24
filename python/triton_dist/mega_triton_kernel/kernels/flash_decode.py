
import triton
import triton.language as tl
from triton.language.extra import libdevice
from .task_context_utils import TaskBaseInfo, Scoreboard
from .utils import tanh


@triton.jit
def attn_gqa_fwd_batch_decode_split_kv_task_para(
    tile_id_or_start, MAX_NUM_TENSOR_DIMS,
    io_tensors_ptr,

    SM_SCALE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,       # 比如 16
    MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    scoreboard_ptr,
    layer_id,
    task_id,
    TILE_READY_SIGNAL: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
):
    tile_id = tile_id_or_start
    q_tensor = task_base_info_get_tensor(io_tensors_ptr, 0, MAX_NUM_TENSOR_DIMS)
    q_ptr = tensor_desc_data_ptr(q_tensor, tl.bfloat16)
    k_cache_tensor = task_base_info_get_tensor(io_tensors_ptr, 1, MAX_NUM_TENSOR_DIMS)
    k_cache_ptr = tensor_desc_data_ptr(k_cache_tensor, tl.bfloat16)
    v_cache_tensor = task_base_info_get_tensor(io_tensors_ptr, 2, MAX_NUM_TENSOR_DIMS)
    v_cache_ptr = tensor_desc_data_ptr(v_cache_tensor, tl.bfloat16)
    block_table_tensor = task_base_info_get_tensor(io_tensors_ptr, 3, MAX_NUM_TENSOR_DIMS)
    block_table_ptr = tensor_desc_data_ptr(block_table_tensor, tl.int32)
    kv_length_tensor = task_base_info_get_tensor(io_tensors_ptr, 4, MAX_NUM_TENSOR_DIMS)
    kv_length_ptr = tensor_desc_data_ptr(kv_length_tensor, tl.int32)
    partial_out_tensor = task_base_info_get_tensor(io_tensors_ptr, 5, MAX_NUM_TENSOR_DIMS)
    partial_out_ptr = tensor_desc_data_ptr(partial_out_tensor, tl.float32)
    lse_tensor = task_base_info_get_tensor(io_tensors_ptr, 6, MAX_NUM_TENSOR_DIMS)
    lse_ptr = tensor_desc_data_ptr(lse_tensor, tl.float32)

    tl.static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0)
    NUM_Q_HEADS_PER_GROUP: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    K_HEAD_DIM: tl.constexpr = Q_HEAD_DIM
    
    # Stride 定义
    stride_q_bs: tl.constexpr = NUM_Q_HEADS * Q_HEAD_DIM
    stride_q_h: tl.constexpr = Q_HEAD_DIM
    stride_table_bs: tl.constexpr = MAX_NUM_BLOCKS_PER_SEQ
    stride_cache_bs: tl.constexpr = NUM_KV_HEADS * K_HEAD_DIM
    stride_cache_h: tl.constexpr = K_HEAD_DIM
    
    stride_o_bs: tl.constexpr = NUM_Q_HEADS * NUM_KV_SPLITS * V_HEAD_DIM
    stride_o_h: tl.constexpr = NUM_KV_SPLITS * V_HEAD_DIM
    stride_o_split: tl.constexpr = V_HEAD_DIM
    stride_lse_bs: tl.constexpr = NUM_Q_HEADS * NUM_KV_SPLITS
    stride_lse_h: tl.constexpr = NUM_KV_SPLITS
    SUB_BLOCK_H: tl.constexpr = BLOCK_H // 2

    # NUM_Q_HEADS = 32
    # NUM_KV_HEADS = 8
    # V_HEAD_DIM = Q_HEAD_DIM = 128
    # NUM_KV_SPLITS = 1
    # BLOCK_H = 4

    # --- 2. 任务分配 ---
    head_blocks = tl.cdiv(NUM_Q_HEADS, BLOCK_H)
    bid = tile_id // (head_blocks * NUM_KV_SPLITS) # seq id
    hid = tile_id % (head_blocks * NUM_KV_SPLITS) // NUM_KV_SPLITS # head group id
    kv_hid = hid // tl.cdiv(NUM_Q_HEADS_PER_GROUP, BLOCK_H) # kv head id
    split_kv_id = tile_id % NUM_KV_SPLITS # split id
    offs_d = tl.arange(0, K_HEAD_DIM)
    
    head_off = hid * BLOCK_H
    cur_head = head_off + tl.arange(0, BLOCK_H) # 当前task要计算的q head id
    offs_q = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_d[None, :] # 加载query的某个seq的某个head group
    q = tl.load(q_ptr + offs_q)

    cur_kv_seq_len = tl.load(kv_length_ptr + bid)
    kv_len_per_split = tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS)
    page_num = tl.cdiv(cur_kv_seq_len, PAGE_SIZE)
    page_per_split = tl.cdiv(page_num, NUM_KV_SPLITS)
    start_logical_block_idx = page_per_split * split_kv_id
    end_logical_block_idx = tl.minimum(start_logical_block_idx + page_per_split, page_num)
    
    SCALAR_STRIDE: tl.constexpr = 8   # 8 * fp32 = 32B
    e_max = tl.full([BLOCK_H * SCALAR_STRIDE], -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H * SCALAR_STRIDE], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, V_HEAD_DIM], dtype=tl.float32)

    offs_page = tl.arange(0, PAGE_SIZE)
    offs_in_page = offs_page[:, None] * stride_cache_bs + offs_d[None, :]

    for logical_block_idx in range(start_logical_block_idx, end_logical_block_idx):
        current_token_indices = logical_block_idx * PAGE_SIZE + offs_page
        mask_valid_tokens = current_token_indices < cur_kv_seq_len

        physical_block_id = tl.load(block_table_ptr + bid * stride_table_bs + logical_block_idx).to(tl.int64)  # 避免大batch下32位地址偏移溢出
        kv_off = physical_block_id * (PAGE_SIZE * stride_cache_bs) + kv_hid * stride_cache_h + offs_in_page
        k_tile = tl.load(k_cache_ptr + kv_off)
        v_tile = tl.load(v_cache_ptr + kv_off)

        qk = tl.dot(q, tl.trans(k_tile), allow_tf32=False) * SM_SCALE
        qk = tl.where(mask_valid_tokens[None, :], qk, float("-inf"))

        re_scale = tl.zeros([BLOCK_H], dtype=tl.float32)                                                                                                                                                    
        p_sub = tl.zeros([BLOCK_H, PAGE_SIZE], dtype=v_tile.dtype)

        for sub_id in al.parallel(0,2,bind_sub_block=True):
            for I in tl.static_range(0, SUB_BLOCK_H):
                i = sub_id * SUB_BLOCK_H + I
                qk_h = al.extract_slice(qk, (i, 0), (1, PAGE_SIZE), (1, 1))
                e_max_h = al.extract_slice(e_max, [i * SCALAR_STRIDE], [1], [1])
                e_sum_h = al.extract_slice(e_sum, [i * SCALAR_STRIDE], [1], [1])

                n_e_max = tl.maximum(tl.max(qk_h, 1), e_max_h)        
                re_scale_h = tl.exp(e_max_h - n_e_max)
                p = tl.exp(qk_h - n_e_max[:, None])
                e_sum_h = e_sum_h * re_scale_h + tl.sum(p, 1)
                
                p_sub    = al.insert_slice(p_sub,    p.to(v_tile.dtype), (i, 0), (1, PAGE_SIZE), (1,1))
                e_max    = al.insert_slice(e_max,    n_e_max,    [i * SCALAR_STRIDE], [1], [1])
                e_sum    = al.insert_slice(e_sum,    e_sum_h,    [i * SCALAR_STRIDE], [1], [1])
                re_scale = al.insert_slice(re_scale, re_scale_h, [i], [1], [1])       

        pv = tl.dot(p_sub, v_tile, allow_tf32=False)

        for sub_id in al.parallel(0,2,bind_sub_block=True):
            for I in tl.static_range(0, SUB_BLOCK_H):
                i = sub_id * SUB_BLOCK_H + I
                pv_h       = al.extract_slice(pv, (i, 0), (1, V_HEAD_DIM), (1, 1))
                acc_h      = al.extract_slice(acc, (i, 0), (1, V_HEAD_DIM), (1, 1))
                re_scale_h = al.extract_slice(re_scale, [i], [1], [1])

                acc_h = acc_h * re_scale_h[:, None] + pv_h
                
                acc = al.insert_slice(acc, acc_h, (i, 0), (1, V_HEAD_DIM), (1, 1))

    with al.scope(core_mode="cube"):
        tl.sync_block_set('cube', 'vector', 13)
    with al.scope(core_mode="vector"):
        tl.sync_block_wait('cube', 'vector', 13)

    for sub_id in al.parallel(0,2,bind_sub_block=True):
        for I in tl.static_range(0, SUB_BLOCK_H):
            i = sub_id * SUB_BLOCK_H + I
            e_max_h  = al.extract_slice(e_max, [i * SCALAR_STRIDE], [1], [1])
            e_sum_h  = al.extract_slice(e_sum, [i * SCALAR_STRIDE], [1], [1])
            acc_h  = al.extract_slice(acc, (i, 0), (1, V_HEAD_DIM), (1, 1)), [V_HEAD_DIM]
            
            e_sum_h = tl.where(e_sum_h > 0.0, e_sum_h, 1.0)
            e_max_h = e_max_h + tl.log(e_sum_h)
            acc_h = acc_h / e_sum_h[None, :]

            offs_out = bid * stride_o_bs + (head_off + i) * stride_o_h + split_kv_id * stride_o_split + offs_d[None, :]
            tl.store(partial_out_ptr + offs_out, acc_h)

            offs_log = bid * stride_lse_bs + (head_off + i) * stride_lse_h + split_kv_id + tl.arange(0, 1)
            tl.store(lse_ptr + offs_log, e_max_h)

    with al.scope(core_mode="cube"):
        tl.sync_block_set('cube', 'vector', 12)
    with al.scope(core_mode="vector"):
        tl.sync_block_wait('cube', 'vector', 12)

    scoreboard_release_tile_flat(
        scoreboard_ptr=scoreboard_ptr, 
        layer_id=layer_id, 
        task_id=task_id,
        tile_id=tile_id_or_start,
        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
        MAX_TASK_ID=MAX_TASK_ID,
        MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
    )