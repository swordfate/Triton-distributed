"""
完全复制 megakernel wrapper: for 循环 + work queue load + linear call
看和独立测试的区别在哪
"""

import torch
import triton
import triton.language as tl
from torch_npu.contrib import transfer_to_npu


@triton.jit
def kernel_with_wrapper(
    work_queues,
    num_tasks_per_wq,
    INT_PER_TASK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    SUB_BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    scoreboard_ptr, task_deps_ptr,
    INT_PER_DEPS: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
    debug_counts,
):
    # ====== 完全复制 megakernel wrapper ======
    sm_id = tl.program_id(axis=0)
    num_tasks = tl.load(num_tasks_per_wq + sm_id)
    offset = INT_PER_TASK * NUM_SMS

    TASK_TYPE_OFFSET = 0
    LAYER_ID_OFFSET = 1
    TASK_ID_OFFSET = 2
    TILE_ID_OR_START_OFFSET = 3
    DEPEND_ENTRY_START_OFFSET = 4
    DEPEND_ENTRY_END_OFFSET = 5
    IO_TENSORS_OFFSET = 6

    for i in range(num_tasks):
        task_type = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TASK_TYPE_OFFSET).to(tl.int32)
        layer_id = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + LAYER_ID_OFFSET).to(tl.int32)
        task_id = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TASK_ID_OFFSET).to(tl.int32)
        tile_id_or_start = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TILE_ID_OR_START_OFFSET).to(tl.int32)
        depend_entry_start = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + DEPEND_ENTRY_START_OFFSET).to(tl.int32)
        depend_entry_end = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + DEPEND_ENTRY_END_OFFSET).to(tl.int32)
        io_tensors_ptr = work_queues + i * offset + sm_id * INT_PER_TASK + IO_TENSORS_OFFSET

        if task_type == 1:
            linear_task_compute_lite(
                tile_id_or_start, MAX_NUM_TENSOR_DIMS, io_tensors_ptr,
                BLOCK_SIZE_M, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES,
                scoreboard_ptr, layer_id, task_id,
                TILE_READY_SIGNAL=tl.constexpr(2),
                MAX_TASK_ID=MAX_TASK_ID,
                MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP,
            )


@triton.jit
def linear_task_compute_lite(
    tile_id, MAX_DIMS,
    io_tensors_ptr,
    BM: tl.constexpr, BN: tl.constexpr,
    SUB: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr,
    scoreboard_ptr, layer_id, task_id,
    TILE_READY_SIGNAL: tl.constexpr,
    MAX_TASK_ID: tl.constexpr, MAX_TILES: tl.constexpr,
):
    # 简化版 tensor_desc (和 megakernel 一样从 work queue 读指针)
    INT_PER_TENSOR = MAX_DIMS + 2

    a_lo = tl.load(io_tensors_ptr)
    a_hi = tl.load(io_tensors_ptr + 1)
    a_ptr = ((a_hi.to(tl.uint64) << 32) | (a_lo.to(tl.uint64) & 0xFFFFFFFF)).to(tl.pointer_type(tl.bfloat16))

    b_lo = tl.load(io_tensors_ptr + INT_PER_TENSOR)
    b_hi = tl.load(io_tensors_ptr + INT_PER_TENSOR + 1)
    b_ptr = ((b_hi.to(tl.uint64) << 32) | (b_lo.to(tl.uint64) & 0xFFFFFFFF)).to(tl.pointer_type(tl.bfloat16))

    c_lo = tl.load(io_tensors_ptr + 2 * INT_PER_TENSOR)
    c_hi = tl.load(io_tensors_ptr + 2 * INT_PER_TENSOR + 1)
    c_ptr = ((c_hi.to(tl.uint64) << 32) | (c_lo.to(tl.uint64) & 0xFFFFFFFF)).to(tl.pointer_type(tl.bfloat16))

    M = tl.load(io_tensors_ptr + 2).to(tl.int32)
    K = tl.load(io_tensors_ptr + 3).to(tl.int32)
    N = tl.load(io_tensors_ptr + 2 * INT_PER_TENSOR + 3).to(tl.int32)

    num_pid_n = tl.cdiv(N, BN)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BM
    base_start_n = pid_n * BN
    offs_am = start_m + tl.arange(0, BM)
    k_tiles = tl.cdiv(K, BK)

    for i in tl.range(0, BN, SUB, num_stages=NS):
        start_n = base_start_n + i
        offs_bn = start_n + tl.arange(0, SUB)
        acc = tl.zeros((BM, SUB), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BK + tl.arange(0, BK)
            a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
            b_ptrs = b_ptr + (offs_bn[:, None] * K + offs_k[None, :])
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b.T, acc)
        offs_cm = pid_m * BM + tl.arange(0, BM)
        offs_cn = start_n + tl.arange(0, SUB)
        c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
        tl.store(c_ptrs, acc.to(tl.bfloat16))


def make_wq(device):
    """构建一个 task 的工作队列条目"""
    M, N, K = 1, 6144, 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    c = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

    INT_PER = 6  # MAX_DIMS(4)+2
    INT_PER_TASK = 6 + 3 * INT_PER  # header(6) + 3 tensors * 6 = 24

    # 填充 header
    wq = torch.zeros(INT_PER_TASK, dtype=torch.int32, device=device)
    wq[0] = 1  # task_type = 1 (QKVProj)
    wq[1] = 0  # layer_id
    wq[2] = 0  # task_id
    wq[3] = 0  # tile_id_or_start
    wq[4] = 0  # deps_l
    wq[5] = 0  # deps_r

    # 填充 tensor desc (3 tensors: a, b, c)
    for idx, t in enumerate([a, b, c]):
        base = 6 + idx * INT_PER
        ptr = t.data_ptr()
        wq[base + 0] = ptr & 0xFFFFFFFF
        wq[base + 1] = (ptr >> 32) & 0xFFFFFFFF
        shape = list(t.shape) + [1] * (4 - len(t.shape))
        for d in range(4):
            wq[base + 2 + d] = shape[d]

    return wq.reshape(1, 1, INT_PER_TASK), a, b, c


if __name__ == "__main__":
    import sys
    device = 'npu' if len(sys.argv) < 2 else sys.argv[1]
    print(f"=== Wrapper test on {device} ===")

    wq, a, b, c = make_wq(device)
    nt = torch.tensor([1], dtype=torch.int32, device=device)  # 1 task
    sb = torch.zeros(100, dtype=torch.int32, device=device)
    td = torch.zeros(0, dtype=torch.int32, device=device).reshape(0, 2)
    dc = torch.zeros(10, dtype=torch.int32, device=device)

    for name, ns in [("ns5", 5), ("ns3", 3), ("ns2", 2), ("ns1", 1)]:
        try:
            kernel_with_wrapper[(1, 1, 1)](
                wq, nt,
                INT_PER_TASK=wq.shape[2], NUM_SMS=1, MAX_NUM_TENSOR_DIMS=4,
                BLOCK_SIZE_M=16, BLOCK_SIZE_N=6240,
                SUB_BLOCK_SIZE_N=416, BLOCK_SIZE_K=256, NUM_STAGES=ns,
                scoreboard_ptr=sb, task_deps_ptr=td,
                INT_PER_DEPS=2, MAX_TASK_ID=10, MAX_NUM_TILES_PER_OP=128,
                debug_counts=dc,
            )
            torch.npu.synchronize()
            print(f"  {name}: OK  c_sum={c.sum().item():.1f}")
        except Exception as e:
            msg = str(e)
            if "507015" in msg:
                print(f"  {name}: AICORE RUNTIME ERROR")
            else:
                print(f"  {name}: {msg[:120]}")
