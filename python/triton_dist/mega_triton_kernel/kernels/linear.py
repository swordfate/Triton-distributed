import triton
import triton.language as tl
from torch_npu.contrib import transfer_to_npu

from .task_context_utils import *
from triton.language.extra.cann import extension as al

@triton.jit
def tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, 
                             M: tl.constexpr, 
                             N: tl.constexpr, 
                             K: tl.constexpr, 
                             BLOCK_SIZE_M: tl.constexpr, 
                             BLOCK_SIZE_N: tl.constexpr,  # 用于减少分核数
                             SUB_BLOCK_SIZE_N: tl.constexpr, # 单次数据加载的大小
                             BLOCK_SIZE_K: tl.constexpr,
                             NUM_STAGES: tl.constexpr
                             ):
    # --- 1. Grid 计算 ---
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    
    start_m = pid_m * BLOCK_SIZE_M
    base_start_n = pid_n * BLOCK_SIZE_N
    
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    # --- 2. 新增的 N 维度切片循环 ---
    # 分批算 SUB_BLOCK_SIZE_N
    for i in tl.range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N, num_stages=NUM_STAGES):
        # start_n = base_start_n + i

        # offs = tl.arange(0, BLOCK_SIZE_M * SUB_BLOCK_SIZE_N)
        # offs_m = offs // SUB_BLOCK_SIZE_N
        # offs_n = offs % SUB_BLOCK_SIZE_N

        # cm = start_m + offs_m
        # cn = start_n + offs_n

        # c_ptrs = c_ptr + cm * N + cn
        # c_mask = (cm < M) & (cn < N)

        # zero = tl.full((BLOCK_SIZE_M * SUB_BLOCK_SIZE_N,), 0.0, dtype=tl.float32)
        # tl.store(c_ptrs, zero.to(c_ptr.dtype.element_ty), mask=c_mask)
        # 计算当前子块的 N 偏移
        current_n_offset = i
        start_n = base_start_n + current_n_offset
        
        # 边界检查：如果 BLOCK_SIZE_N 不是 N 的整数倍，或者 SUB_BLOCK 溢出，需要掩码保护
        offs_bn = start_n + tl.arange(0, SUB_BLOCK_SIZE_N)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, SUB_BLOCK_SIZE_N), SUB_BLOCK_SIZE_N)
        
        # 初始化当前子块的累加器 [BLOCK_M, SUB_BLOCK_N]
        accumulator = tl.zeros((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), dtype=tl.float32)
        
        # --- 3. 原有的 K 循环 (计算的核心) ---
        for ki in tl.range(0, k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
            b_ptrs = b_ptr + (offs_bn[:, None] * K + offs_k[None, :])
            
            k_mask = offs_k < K
            a_mask = (offs_am[:, None] < M) & (k_mask[None, :])
            b_mask = (offs_bn[:, None] < N) & (k_mask[None, :])
            
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            # [M, K] x [N_sub, K].T -> [M, N_sub]
            accumulator = tl.dot(a, b.T, accumulator)
        
        # --- 4. 存储当前子块的结果 ---
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = start_n + tl.arange(0, SUB_BLOCK_SIZE_N) # 使用当前的 start_n
        c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(c_ptr.dtype.element_ty)
        
        # with al.scope(core_mode="cube"):
        #     tl.sync_block_set('cube', 'vector', 13)
        # with al.scope(core_mode="vector"):
        #     tl.sync_block_wait('cube', 'vector', 13)
        
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def tile_wise_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, 
    SUB_BLOCK_SIZE_N: tl.constexpr, # 单次数据加载的大小
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    主kernel，调用tile_wise_matmul_compute作为子函数
    """
    # 获取当前处理的tile ID
    tile_id = tl.program_id(axis=0)
    
    # 调用原始的子函数实现
    tile_wise_matmul_compute(
        tile_id, 
        a_ptr, b_ptr, c_ptr, 
        M, N, K, 
        BLOCK_SIZE_M, 
        BLOCK_SIZE_N,
        SUB_BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        NUM_STAGES,
    )
    


@triton.jit
def tile_range_matmul_compute_and_notify(tile_start, sb_base_ptr, a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M,
                                         BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES, TILE_READY_SIGNAL, NUM_SMS):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    for tile_id in tl.range(tile_start, num_tiles, NUM_SMS, flatten=True, warp_specialize=True):
        tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                 NUM_STAGES)
        st(sb_base_ptr + tile_id, TILE_READY_SIGNAL, "gpu", "release")


@triton.jit
def linear_task_compute(
                        tile_id_or_start, 
                        io_tensors_ptr,
                        MAX_NUM_TENSOR_DIMS: tl.constexpr,
                        BLOCK_SIZE_M: tl.constexpr,
                        BLOCK_SIZE_N: tl.constexpr, 
                        SUB_BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr, NUM_STAGES: tl.constexpr,
                        ALIGNMENT_K: tl.constexpr,
                        scoreboard_ptr,
                        layer_id,
                        task_id,
                        TILE_READY_SIGNAL: tl.constexpr,
                        MAX_TASK_ID: tl.constexpr,
                        MAX_NUM_TILES_PER_OP: tl.constexpr,
                        ):

    input = task_base_info_get_tensor(io_tensors_ptr, 0, MAX_NUM_TENSOR_DIMS)
    weight = task_base_info_get_tensor(io_tensors_ptr, 1, MAX_NUM_TENSOR_DIMS)
    output = task_base_info_get_tensor(io_tensors_ptr, 2, MAX_NUM_TENSOR_DIMS)

    M = tensor_desc_size(input, 0)
    K = tensor_desc_size(input, 1, ALIGNMENT_K)
    N = tensor_desc_size(weight, 0)

    a_ptr = tensor_desc_data_ptr(input, tl.bfloat16)
    b_ptr = tensor_desc_data_ptr(weight, tl.bfloat16)
    c_ptr = tensor_desc_data_ptr(output, tl.bfloat16)

    tile_id = tile_id_or_start
    tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K, 
                             BLOCK_SIZE_M, 
                             BLOCK_SIZE_N, 
                             SUB_BLOCK_SIZE_N,
                             BLOCK_SIZE_K,
                             NUM_STAGES
                             )

    with al.scope(core_mode="cube"):
        tl.sync_block_set('cube', 'vector', 10)
    with al.scope(core_mode="vector"):
        tl.sync_block_wait('cube', 'vector', 10)
    
    scoreboard_release_tile_flat(
        scoreboard_ptr=scoreboard_ptr, 
        layer_id=layer_id, 
        task_id=task_id,
        tile_id=tile_id_or_start,
        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
        MAX_TASK_ID=MAX_TASK_ID,
        MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
    )