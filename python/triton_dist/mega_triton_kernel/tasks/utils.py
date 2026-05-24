from typing import List, Tuple
import torch


def build_tile_desc(full_shape: List[int], tile_sizes: List[int], tile_id: int,
                    return_valid_size=False) -> Tuple[List[int], List[int]]:
    num_tiles = []
    ndim = len(full_shape)
    assert len(tile_sizes) == ndim
    for x, y in zip(full_shape, tile_sizes):
        num_tiles.append(cdiv(x, y))
    strides = [1 for i in range(len(num_tiles))]
    for j in range(ndim - 1, 0, -1):
        strides[j - 1] *= strides[j]
    start_indices = []
    data_sizes = []
    for i in range(0, ndim):
        v = tile_id // strides[i]
        tile_id = tile_id % strides[i]
        st = v * tile_sizes[i]
        start_indices.append(st)
        valid_size = tile_sizes[i]
        if return_valid_size:
            valid_size = min(full_shape[i] - st, valid_size)
        data_sizes.append(valid_size)
    return start_indices, data_sizes


def cdiv(x: int, y: int) -> int:
    assert x > 0 and y > 0
    return (x + y - 1) // y


def torch_dtype_to_triton_dtype_str(torch_dtype):
    DTYPE_MAP = {
        torch.float16: "tl.float16",
        torch.bfloat16: "tl.bfloat16",
        torch.float: "tl.float32",
        torch.int32: "tl.int32",
    }
    assert torch_dtype in DTYPE_MAP, f"{torch_dtype} is not in DTYPE_MAP"
    return DTYPE_MAP[torch_dtype]