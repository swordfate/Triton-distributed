import torch
import triton
import triton.language as tl
import math
from torch_npu.contrib import transfer_to_npu
from .task_context_utils import *


@triton.jit
def rmsnorm_rope_update_kv_cache_task_compute(
    tile_id_or_start, MAX_NUM_TENSOR_DIMS,
    io_tensors_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,
    Q_RMS_EPS: tl.constexpr,
    K_RMS_EPS: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    scoreboard_ptr,
    layer_id,
    task_id,
    TILE_READY_SIGNAL: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
):
    qkv_tensor = task_base_info_get_tensor(io_tensors_ptr, 0, MAX_NUM_TENSOR_DIMS)
    block_tables_tensor = task_base_info_get_tensor(io_tensors_ptr, 1, MAX_NUM_TENSOR_DIMS)
    kv_lens_tensor = task_base_info_get_tensor(io_tensors_ptr, 2, MAX_NUM_TENSOR_DIMS)
    q_rms_weight_tensor = task_base_info_get_tensor(io_tensors_ptr, 3, MAX_NUM_TENSOR_DIMS)
    k_rms_weight_tensor = task_base_info_get_tensor(io_tensors_ptr, 4, MAX_NUM_TENSOR_DIMS)
    cos_cache_tensor = task_base_info_get_tensor(io_tensors_ptr, 5, MAX_NUM_TENSOR_DIMS)
    sin_cache_tensor = task_base_info_get_tensor(io_tensors_ptr, 6, MAX_NUM_TENSOR_DIMS)
    k_cache_tensor = task_base_info_get_tensor(io_tensors_ptr, 7, MAX_NUM_TENSOR_DIMS)
    v_cache_tensor = task_base_info_get_tensor(io_tensors_ptr, 8, MAX_NUM_TENSOR_DIMS)
    q_norm_rope_tensor = task_base_info_get_tensor(io_tensors_ptr, 9, MAX_NUM_TENSOR_DIMS)

    # num tiles of qkv
    tile_id = tile_id_or_start

    k_cache_ptr = tensor_desc_data_ptr(k_cache_tensor, tl.bfloat16)
    v_cache_ptr = tensor_desc_data_ptr(v_cache_tensor, tl.bfloat16)

    q_rms_weight_ptr = tensor_desc_data_ptr(q_rms_weight_tensor, tl.bfloat16)
    k_rms_weight_ptr = tensor_desc_data_ptr(k_rms_weight_tensor, tl.bfloat16)

    sin_cos_batch = tensor_desc_size(cos_cache_tensor, 0)
    seq_len = tensor_desc_size(qkv_tensor, 1)
    qkv_ptr = tensor_desc_data_ptr(qkv_tensor, tl.bfloat16)
    q_norm_rope_ptr = tensor_desc_data_ptr(q_norm_rope_tensor, tl.bfloat16)
    sin_ptr = tensor_desc_data_ptr(sin_cache_tensor, tl.float32)
    cos_ptr = tensor_desc_data_ptr(cos_cache_tensor, tl.float32)
    kv_lens_ptr = tensor_desc_data_ptr(kv_lens_tensor, tl.int32)
    block_table_ptr = tensor_desc_data_ptr(block_tables_tensor, tl.int32)


    # --- 以下为几乎完全保留的原始逻辑 ---
    tl.static_assert(Q_HEAD_DIM == V_HEAD_DIM)
    num_total_heads: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS * 2
    num_qk_heads: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS

    q_head_dim: tl.constexpr = Q_HEAD_DIM
    # PADDED_Q_HEAD_DIM: tl.constexpr = next_power_of_2(Q_HEAD_DIM)
    # PADDED_V_NUM_HEADS: tl.constexpr = next_power_of_2(NUM_KV_HEADS)
    # 展开 next_power_of_2
    PADDED_Q_HEAD_DIM: tl.constexpr = Q_HEAD_DIM
    PADDED_Q_HEAD_DIM1: tl.constexpr = PADDED_Q_HEAD_DIM - 1
    PADDED_Q_HEAD_DIM2: tl.constexpr = PADDED_Q_HEAD_DIM1 | (PADDED_Q_HEAD_DIM1 >> 1)
    PADDED_Q_HEAD_DIM3: tl.constexpr = PADDED_Q_HEAD_DIM2 | (PADDED_Q_HEAD_DIM2 >> 2)
    PADDED_Q_HEAD_DIM4: tl.constexpr = PADDED_Q_HEAD_DIM3 | (PADDED_Q_HEAD_DIM3 >> 4)
    PADDED_Q_HEAD_DIM5: tl.constexpr = PADDED_Q_HEAD_DIM4 | (PADDED_Q_HEAD_DIM4 >> 8)
    PADDED_Q_HEAD_DIM6: tl.constexpr = PADDED_Q_HEAD_DIM5 | (PADDED_Q_HEAD_DIM5 >> 16)
    PADDED_Q_HEAD_DIM7: tl.constexpr = PADDED_Q_HEAD_DIM6 | (PADDED_Q_HEAD_DIM6 >> 32)
    PADDED_Q_HEAD_DIM8: tl.constexpr = PADDED_Q_HEAD_DIM7 + 1
    PADDED_V_NUM_HEADS: tl.constexpr = NUM_KV_HEADS
    PADDED_V_NUM_HEADS1: tl.constexpr = PADDED_V_NUM_HEADS - 1
    PADDED_V_NUM_HEADS2: tl.constexpr = PADDED_V_NUM_HEADS1 | (PADDED_V_NUM_HEADS1 >> 1)
    PADDED_V_NUM_HEADS3: tl.constexpr = PADDED_V_NUM_HEADS2 | (PADDED_V_NUM_HEADS2 >> 2)
    PADDED_V_NUM_HEADS4: tl.constexpr = PADDED_V_NUM_HEADS3 | (PADDED_V_NUM_HEADS3 >> 4)
    PADDED_V_NUM_HEADS5: tl.constexpr = PADDED_V_NUM_HEADS4 | (PADDED_V_NUM_HEADS4 >> 8)
    PADDED_V_NUM_HEADS6: tl.constexpr = PADDED_V_NUM_HEADS5 | (PADDED_V_NUM_HEADS5 >> 16)
    PADDED_V_NUM_HEADS7: tl.constexpr = PADDED_V_NUM_HEADS6 | (PADDED_V_NUM_HEADS6 >> 32)
    PADDED_V_NUM_HEADS8: tl.constexpr = PADDED_V_NUM_HEADS7 + 1

    K_HEAD_DIM: tl.constexpr = Q_HEAD_DIM

    cols = tl.arange(0, PADDED_Q_HEAD_DIM8)
    offs_vh = tl.arange(0, PADDED_V_NUM_HEADS8)

    stride_table_bs: tl.constexpr = MAX_NUM_BLOCKS_PER_SEQ
    stride_value_per_token: tl.constexpr = V_HEAD_DIM * NUM_KV_HEADS
    stride_qkv_batch = seq_len * num_total_heads * V_HEAD_DIM
    stride_qkv_token: tl.constexpr = V_HEAD_DIM * num_total_heads
    stride_key_per_token: tl.constexpr = K_HEAD_DIM * NUM_KV_HEADS

    idx_0 = tile_id // num_qk_heads
    idx_1 = tile_id % num_qk_heads

    batch_size = tensor_desc_size(qkv_tensor, 0)
    batch_grp_idx = idx_0 // seq_len
    seq_idx = idx_0 % seq_len
    cos_row_idx = seq_idx

    offs_b = tl.arange(0, BLOCK_SIZE_B)
    batch_idx = batch_grp_idx * BLOCK_SIZE_B + offs_b
    batch_mask = batch_idx < batch_size

    history_kv = tl.load(kv_lens_ptr + batch_idx, mask=batch_mask, other=seq_len) - seq_len
    global_seq_id = seq_idx + history_kv
    page_id = global_seq_id // PAGE_SIZE
    page_off = global_seq_id - page_id * PAGE_SIZE

    block_entry_id = tl.load(block_table_ptr + batch_idx * stride_table_bs + page_id, mask=batch_mask, other=0)
    cache_token_idx = block_entry_id * PAGE_SIZE + page_off
    
    # set value_cache in tile which calculate key
    if idx_1 >= NUM_Q_HEADS:
        key_head_idx = idx_1 - NUM_Q_HEADS

        v_load_ptrs = (
            qkv_ptr
            + batch_idx[:, None] * stride_qkv_batch
            + seq_idx * stride_qkv_token
            + num_qk_heads * Q_HEAD_DIM
            + key_head_idx * V_HEAD_DIM
            + cols[None, :]
        )

        v_mask = batch_mask[:, None] & (cols[None, :] < V_HEAD_DIM)
        value_tile = tl.load(v_load_ptrs,mask=v_mask,other=0.0)

        v_store_ptrs = (
            v_cache_ptr
            + cache_token_idx[:, None] * stride_value_per_token
            + key_head_idx * V_HEAD_DIM
            + cols[None, :]
        )

        tl.store(v_store_ptrs,value_tile,mask=v_mask)


    x_ptrs = (
        qkv_ptr
        + batch_idx[:, None] * stride_qkv_batch
        + seq_idx * stride_qkv_token
        + idx_1 * q_head_dim
        + cols[None, :]
    )

    x_mask = batch_mask[:, None] & (cols[None, :] < q_head_dim)

    if idx_1 >= NUM_Q_HEADS:
        RMS_EPS = K_RMS_EPS
    else:
        RMS_EPS = Q_RMS_EPS

    # 上面的写法会报错
    
    a = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

    # a shape: [BLOCK_SIZE_B, PADDED_Q_HEAD_DIM8]
    # rms shape: [BLOCK_SIZE_B]
    rms = tl.rsqrt(tl.sum(a * a, axis=1) / q_head_dim + RMS_EPS)

    mask = cols < q_head_dim

    half_dim: tl.constexpr = PADDED_Q_HEAD_DIM8 // 2

    cos_offsets = tl.arange(0, half_dim)

    cos_mask = (batch_mask[:, None] & (cos_offsets[None, :] < q_head_dim // 2))

    cos_seq_len = tensor_desc_size(cos_cache_tensor, 1)

    cos_batch_offset = (
        batch_idx[:, None]
        * cos_seq_len
        * q_head_dim
        * (sin_cos_batch != 1)
    )

    pos_offset = (global_seq_id[:, None] * q_head_dim + cos_offsets[None, :])

    cos_row = tl.load(cos_ptr + cos_batch_offset + pos_offset, mask=cos_mask, other=0.0)
    sin_row = tl.load(sin_ptr + cos_batch_offset + pos_offset, mask=cos_mask, other=0.0)

    first_half_qk_offsets = tl.arange(0, half_dim)
    second_half_qk_offsets = first_half_qk_offsets + (q_head_dim // 2)

    qk_mask = first_half_qk_offsets < q_head_dim // 2
    qk_mask_b = batch_mask[:, None] & qk_mask[None, :]

    if idx_1 >= NUM_Q_HEADS:
        w = tl.load(k_rms_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = w[None, :] * (a * rms[:, None])
        y0 = tl.extract_slice(y, (0, 0), (BLOCK_SIZE_B, half_dim), (1, 1)).to(tl.bfloat16)
        y1 = tl.extract_slice(y, (0, half_dim), (BLOCK_SIZE_B, half_dim), (1, 1)).to(tl.bfloat16)

        new_qkv_tile_0 = y0 * cos_row - y1 * sin_row
        new_qkv_tile_1 = y1 * cos_row + y0 * sin_row

        key_head_idx = idx_1 - NUM_Q_HEADS
        base_k_cache_ptrs = k_cache_ptr + cache_token_idx[:, None] * stride_key_per_token + key_head_idx * K_HEAD_DIM

        tl.store(base_k_cache_ptrs + first_half_qk_offsets[None, :], new_qkv_tile_0, mask=qk_mask_b)
        tl.store(base_k_cache_ptrs + second_half_qk_offsets[None, :], new_qkv_tile_1, mask=qk_mask_b)
    else:
        w = tl.load(q_rms_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = w[None, :] * (a * rms[:, None])
        y0 = tl.extract_slice(y, (0, 0), (BLOCK_SIZE_B, half_dim), (1, 1)).to(tl.bfloat16)
        y1 = tl.extract_slice(y, (0, half_dim), (BLOCK_SIZE_B, half_dim), (1, 1)).to(tl.bfloat16)

        new_qkv_tile_0 = y0 * cos_row - y1 * sin_row
        new_qkv_tile_1 = y1 * cos_row + y0 * sin_row

        stride_q_out_batch = seq_len * NUM_Q_HEADS * Q_HEAD_DIM
        stride_q_out_token: tl.constexpr = NUM_Q_HEADS * Q_HEAD_DIM

        q_out_ptrs = (
            q_norm_rope_ptr
            + batch_idx[:, None] * stride_q_out_batch
            + seq_idx * stride_q_out_token
            + idx_1 * Q_HEAD_DIM
        )
        tl.store(q_out_ptrs + first_half_qk_offsets[None, :], new_qkv_tile_0, mask=qk_mask_b)
        tl.store(q_out_ptrs + second_half_qk_offsets[None, :], new_qkv_tile_1, mask=qk_mask_b)

    scoreboard_release_tile_flat(
        scoreboard_ptr=scoreboard_ptr, 
        layer_id=layer_id, 
        task_id=task_id,
        tile_id=tile_id_or_start,
        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
        MAX_TASK_ID=MAX_TASK_ID,
        MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
    )



@triton.jit
def rmsnorm_task_compute(
    tile_id_or_start,
    io_tensors_ptr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    scoreboard_ptr,
    layer_id,
    task_id,
    TILE_READY_SIGNAL: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
):
    tile_id = tile_id_or_start
    row = tile_id
    input_tensor = task_base_info_get_tensor(io_tensors_ptr, 0, MAX_NUM_TENSOR_DIMS)
    weight_tensor = task_base_info_get_tensor(io_tensors_ptr, 1, MAX_NUM_TENSOR_DIMS)
    output_tensor = task_base_info_get_tensor(io_tensors_ptr, 2, MAX_NUM_TENSOR_DIMS)
    input_ptr = tensor_desc_data_ptr(input_tensor, tl.bfloat16)
    weight_ptr = tensor_desc_data_ptr(weight_tensor, tl.bfloat16)
    output_ptr = tensor_desc_data_ptr(output_tensor, tl.bfloat16)
    N = tensor_desc_size(output_tensor, 1, 16)

    Y = output_ptr + row * N
    X = input_ptr + row * N

    square_sum = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        square_sum += x * x
    rms = tl.rsqrt(tl.sum(square_sum) / N + RMS_EPS)

    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        w = tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        y = w * (x * rms)
        tl.store(Y + cols, y, mask=mask)

    
    scoreboard_release_tile_flat(
        scoreboard_ptr=scoreboard_ptr, 
        layer_id=layer_id, 
        task_id=task_id,
        tile_id=tile_id_or_start,
        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
        MAX_TASK_ID=MAX_TASK_ID,
        MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
    )