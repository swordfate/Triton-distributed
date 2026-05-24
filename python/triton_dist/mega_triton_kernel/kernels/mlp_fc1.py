import triton
import triton.language as tl
from .linear import tile_wise_matmul_compute

from .task_context_utils import *

@triton.jit
def fc1_task_compute(
                    tile_id_or_start, 
                    io_tensors_ptr,
                    MAX_NUM_TENSOR_DIMS: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr, 
                    SUB_BLOCK_SIZE_N: tl.constexpr, # 单次数据加载的大小
                    BLOCK_SIZE_K: tl.constexpr, NUM_STAGES: tl.constexpr,
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
    K = tensor_desc_size(input, 1, 16)
    N = tensor_desc_size(weight, 0)

    a_ptr = tensor_desc_data_ptr(input, tl.bfloat16)
    b_ptr = tensor_desc_data_ptr(weight, tl.bfloat16)
    c_ptr = tensor_desc_data_ptr(output, tl.bfloat16)

    tile_id = tile_id_or_start
    tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K, 
            BLOCK_SIZE_M, 
            BLOCK_SIZE_N, 
            SUB_BLOCK_SIZE_N, # 单次数据加载的大小
            BLOCK_SIZE_K,
            NUM_STAGES
            )
    with al.scope(core_mode="cube"):
        tl.sync_block_set('cube', 'vector', 11)
    with al.scope(core_mode="vector"):
        tl.sync_block_wait('cube', 'vector', 11)
                             
    scoreboard_release_tile_flat(
        scoreboard_ptr=scoreboard_ptr, 
        layer_id=layer_id, 
        task_id=task_id,
        tile_id=tile_id_or_start,
        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
        MAX_TASK_ID=MAX_TASK_ID,
        MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
    )