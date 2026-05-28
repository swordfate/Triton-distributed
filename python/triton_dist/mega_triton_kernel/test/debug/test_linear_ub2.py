"""
测试: 模拟 megakernel 的 task_context 模式, 运行时维度 (非 constexpr),
      看是否触发 CCU 错误.

关键区别: 独立测试 M/N/K 是 tl.constexpr, megakernel 里从 tensor descriptor
         运行时加载 → 编译器 UB/Cube 分配不同.
"""

import torch
import triton
import triton.language as tl
from torch_npu.contrib import transfer_to_npu

try:
    from triton.language.extra.cann import extension as al
except ImportError:
    al = None


# ─── 模拟 megakernel 的 TaskBaseInfo/tensor_desc 模式 ───
@triton.jit
def tensor_desc_data_ptr(base_ptr, dtype):
    lo = tl.load(base_ptr)
    hi = tl.load(base_ptr + 1)
    val_lo = lo.to(tl.uint64) & 0xFFFFFFFF
    val_hi = hi.to(tl.uint64)
    return (val_hi << 32 | val_lo).to(tl.pointer_type(dtype))


@triton.jit
def tensor_desc_size(base_ptr, i):
    return tl.load(base_ptr + i + 2).to(tl.int32)


@triton.jit
def linear_with_runtime_dims(
    io_tensors_ptr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    SUB_BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    scoreboard_ptr, layer_id, task_id,
    TILE_READY_SIGNAL: tl.constexpr, MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
):
    """完全复制 megakernel 中 linear_task_compute 的调用方式:
       所有维度从 io_tensors_ptr 运行时加载, 非 constexpr."""

    # 加载 tensor 描述符
    INT_PER_TENSOR = MAX_NUM_TENSOR_DIMS + 2
    a_tensor = io_tensors_ptr
    b_tensor = io_tensors_ptr + INT_PER_TENSOR
    c_tensor = io_tensors_ptr + 2 * INT_PER_TENSOR

    a_ptr = tensor_desc_data_ptr(a_tensor, tl.bfloat16)
    b_ptr = tensor_desc_data_ptr(b_tensor, tl.bfloat16)
    c_ptr = tensor_desc_data_ptr(c_tensor, tl.bfloat16)

    M = tensor_desc_size(a_tensor, 0)
    K = tensor_desc_size(a_tensor, 1)
    N = tensor_desc_size(c_tensor, 1)

    # --- 下面是 tile_wise_matmul_compute 的核心逻辑 ---
    tile_id = 0  # 单 tile
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    base_start_n = pid_n * BLOCK_SIZE_N
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for i in tl.range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N,
                       num_stages=NUM_STAGES):
        start_n = base_start_n + i
        offs_bn = start_n + tl.arange(0, SUB_BLOCK_SIZE_N)
        accumulator = tl.zeros((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
            b_ptrs = b_ptr + (offs_bn[:, None] * K + offs_k[None, :])
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b.T, accumulator)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = start_n + tl.arange(0, SUB_BLOCK_SIZE_N)
        c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
        tl.store(c_ptrs, accumulator.to(tl.bfloat16))


def make_io_tensors(device):
    """模拟 work queue 中 tensor descriptor 的编码: [ptr_lo, ptr_hi, d0, d1, d2, d3]"""
    M, N, K = 1, 6144, 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    c = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

    INT_PER = 6  # MAX_NUM_TENSOR_DIMS(4) + 2
    io = torch.zeros(3 * INT_PER, dtype=torch.int32, device=device)

    for idx, t in enumerate([a, b, c]):
        base = idx * INT_PER
        ptr = t.data_ptr()
        io[base + 0] = ptr & 0xFFFFFFFF          # ptr_lo
        io[base + 1] = (ptr >> 32) & 0xFFFFFFFF  # ptr_hi
        shape = list(t.shape) + [1] * (4 - len(t.shape))
        for d in range(4):
            io[base + 2 + d] = shape[d]

    return io, a, b, c


if __name__ == "__main__":
    import sys
    device = 'npu' if len(sys.argv) < 2 else sys.argv[1]
    print(f"=== Runtime dimensions test on {device} ===")

    io_t, a, b, c = make_io_tensors(device)

    sb = torch.zeros(100, dtype=torch.int32, device=device)

    for name, sub, bn, ns, dm in [
        ("sub416_bn6240_ns5   ", 416, 6240, 5, False),
        ("sub416_bn6240_ns1   ", 416, 6240, 1, False),
        ("sub416_bn6240_ns1_DM", 416, 6240, 1, True),
    ]:
        try:
            linear_with_runtime_dims[(1, 1, 1)](
                io_t, MAX_NUM_TENSOR_DIMS=4,
                BLOCK_SIZE_M=16, BLOCK_SIZE_N=bn,
                SUB_BLOCK_SIZE_N=sub, BLOCK_SIZE_K=256,
                NUM_STAGES=ns,
                scoreboard_ptr=sb, layer_id=0, task_id=0,
                TILE_READY_SIGNAL=2, MAX_TASK_ID=10,
                MAX_NUM_TILES_PER_OP=128,
            )
            torch.npu.synchronize()
            print(f"  {name}: OK  c_sum={c.sum().item():.1f}")
        except Exception as e:
            msg = str(e)
            if "507015" in msg or "aicore" in msg:
                print(f"  {name}: AICORE RUNTIME ERROR")
            elif "SIGSEGV" in msg:
                print(f"  {name}: SIGSEGV")
            else:
                print(f"  {name}: {msg[:120]}")
