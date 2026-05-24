
import triton
import triton.language as tl
from torch_npu.contrib import transfer_to_npu

from .task_context_utils import *

@triton.jit
def act_mul_up_tile_compute(tile_id, input, output, M, N, ACT_FN, BLOCK_SIZE_M: tl.constexpr,
                            BLOCK_SIZE_N: tl.constexpr):
    tl.static_assert(ACT_FN == tl.constexpr("silu"))

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)  
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    gate_ptrs = input + (offs_m[:, None] * (2 * N) + offs_n[None, :])
    up_ptrs   = input + (offs_m[:, None] * (2 * N) + (offs_n + N)[None, :])

    gate = tl.load(gate_ptrs, mask=mask, other=0.0)
    up   = tl.load(up_ptrs,   mask=mask, other=0.0)

    if ACT_FN == tl.constexpr("silu"):
        gate = gate.to(tl.float32)
        gate = gate * (1.0 / (1.0 + tl.exp((-gate))))
        gate = gate.to(gate_ptrs.dtype.element_ty)
    ret = gate * up
    ret = ret.to(output.dtype.element_ty)
    out_ptrs = output + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(out_ptrs, ret, mask=mask)

@triton.jit
def silu_mul_up_task_compute(
                             tile_id_or_start, 
                             io_tensors_ptr,
                             MAX_NUM_TENSOR_DIMS: tl.constexpr,
                             BLOCK_SIZE_M: tl.constexpr,
                             BLOCK_SIZE_N: tl.constexpr,
                             scoreboard_ptr,
                             layer_id,
                             task_id,
                             TILE_READY_SIGNAL: tl.constexpr,
                             MAX_TASK_ID: tl.constexpr,
                             MAX_NUM_TILES_PER_OP: tl.constexpr,
                             ):
    input = task_base_info_get_tensor(io_tensors_ptr, 0, MAX_NUM_TENSOR_DIMS)
    output = task_base_info_get_tensor(io_tensors_ptr, 1, MAX_NUM_TENSOR_DIMS)

    M = tensor_desc_size(output, 0, 16)
    N = tensor_desc_size(output, 1, 16)

    ACT_FN: tl.constexpr = tl.constexpr("silu")
    a_ptr = tensor_desc_data_ptr(input, tl.bfloat16)
    b_ptr = tensor_desc_data_ptr(output, tl.bfloat16)

    act_mul_up_tile_compute(tile_id_or_start, a_ptr, b_ptr, M, N, ACT_FN, BLOCK_SIZE_M, BLOCK_SIZE_N)
    
    
    scoreboard_release_tile_flat(
        scoreboard_ptr, 
        layer_id=layer_id, 
        task_id=task_id,
        tile_id=tile_id_or_start,
        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
        MAX_TASK_ID=MAX_TASK_ID,
        MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
    )