"""
完全复制 megakernel wrapper + linear_task_compute: 全部内联, 无 jit 子函数调用
"""

import torch, triton, triton.language as tl
from torch_npu.contrib import transfer_to_npu


@triton.jit
def megakernel_minimal(
    work_queues, num_tasks_per_wq,
    scoreboard_ptr, task_deps_ptr,
    INT_PER_DEPS: tl.constexpr, INT_PER_TASK: tl.constexpr,
    MAX_TASK_ID: tl.constexpr, MAX_NUM_TILES_PER_OP: tl.constexpr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr, NUM_SMS: tl.constexpr,
    num_warps: tl.constexpr, debug_counts,
    # ---- linear kernel 参数 ----
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    SUB_BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    TILE_READY_SIGNAL: tl.constexpr = 2
    sm_id = tl.program_id(axis=0)
    num_tasks = tl.load(num_tasks_per_wq + sm_id)
    offset = INT_PER_TASK * NUM_SMS

    TASK_TYPE_OFFSET, LAYER_ID_OFFSET, TASK_ID_OFFSET = 0, 1, 2
    TILE_ID_OR_START_OFFSET = 3
    DEPEND_ENTRY_START_OFFSET, DEPEND_ENTRY_END_OFFSET = 4, 5
    IO_TENSORS_OFFSET = 6

    for i in range(num_tasks):
        task_type = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + TASK_TYPE_OFFSET).to(tl.int32)
        layer_id  = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + LAYER_ID_OFFSET).to(tl.int32)
        task_id   = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + TASK_ID_OFFSET).to(tl.int32)
        tile_id_or_start = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + TILE_ID_OR_START_OFFSET).to(tl.int32)
        deps_start = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + DEPEND_ENTRY_START_OFFSET).to(tl.int32)
        deps_end   = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + DEPEND_ENTRY_END_OFFSET).to(tl.int32)
        io_tensors_ptr = work_queues + i*offset + sm_id*INT_PER_TASK + IO_TENSORS_OFFSET

        if task_type == 1:
            # ─── 内联 linear_task_compute (同 megakernel 的 kernels_for_npu 版本) ───
            MAX_DIMS = MAX_NUM_TENSOR_DIMS
            IPT = MAX_DIMS + 2

            # 加载 tensor desc: a, b, c 的数据指针和维度
            a_lo = tl.load(io_tensors_ptr + 0*IPT + 0)
            a_hi = tl.load(io_tensors_ptr + 0*IPT + 1)
            a_ptr = ((a_hi.to(tl.uint64) << 32) | (a_lo.to(tl.uint64) & 0xFFFFFFFF)).to(tl.pointer_type(tl.bfloat16))

            b_lo = tl.load(io_tensors_ptr + 1*IPT + 0)
            b_hi = tl.load(io_tensors_ptr + 1*IPT + 1)
            b_ptr = ((b_hi.to(tl.uint64) << 32) | (b_lo.to(tl.uint64) & 0xFFFFFFFF)).to(tl.pointer_type(tl.bfloat16))

            c_lo = tl.load(io_tensors_ptr + 2*IPT + 0)
            c_hi = tl.load(io_tensors_ptr + 2*IPT + 1)
            c_ptr = ((c_hi.to(tl.uint64) << 32) | (c_lo.to(tl.uint64) & 0xFFFFFFFF)).to(tl.pointer_type(tl.bfloat16))

            M = tl.load(io_tensors_ptr + 0*IPT + 2).to(tl.int32)  # a.shape[0]
            K = tl.load(io_tensors_ptr + 0*IPT + 3).to(tl.int32)  # a.shape[1]
            N = tl.load(io_tensors_ptr + 2*IPT + 3).to(tl.int32)  # c.shape[1]

            tile_id = tile_id_or_start

            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            start_m = pid_m * BLOCK_SIZE_M
            base_start_n = pid_n * BLOCK_SIZE_N

            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

            for sub_i in tl.range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N, num_stages=NUM_STAGES):
                start_n = base_start_n + sub_i
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

            # scoreboard release
            sb_layer = layer_id * MAX_TASK_ID * MAX_NUM_TILES_PER_OP
            sb_ptr = scoreboard_ptr + sb_layer + task_id * MAX_NUM_TILES_PER_OP
            tl.store(sb_ptr + tile_id, TILE_READY_SIGNAL)


def make_wq(device):
    M, N, K = 1, 6144, 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    c = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

    IP = 6  # MAX_DIMS(4)+2
    IPT = 6 + 3*IP  # header(6) + 3 tensors*6 = 24
    wq = torch.zeros(IPT, dtype=torch.int32, device=device)
    wq[0] = 1  # task_type

    for idx, t in enumerate([a, b, c]):
        base = 6 + idx*IP
        ptr = t.data_ptr()
        wq[base+0] = ptr & 0xFFFFFFFF
        wq[base+1] = (ptr >> 32) & 0xFFFFFFFF
        shape = list(t.shape) + [1]*(4-len(t.shape))
        for d in range(4):
            wq[base+2+d] = shape[d]

    return wq.reshape(1, 1, IPT), a, b, c


if __name__ == "__main__":
    device = 'npu'
    print(f"=== Minimal megakernel test ===")

    wq, a, b, c = make_wq(device)
    nt = torch.tensor([1], dtype=torch.int32, device=device)
    sb = torch.zeros(1024, dtype=torch.int32, device=device)
    td = torch.zeros(0, dtype=torch.int32, device=device).reshape(0, 2)
    dc = torch.zeros(10, dtype=torch.int32, device=device)

    for name, ns in [("ns5", 5), ("ns3", 3), ("ns2", 2), ("ns1", 1)]:
        try:
            megakernel_minimal[(1, 1, 1)](
                wq, nt, sb, td,
                INT_PER_DEPS=2, INT_PER_TASK=wq.shape[2],
                MAX_TASK_ID=10, MAX_NUM_TILES_PER_OP=128,
                MAX_NUM_TENSOR_DIMS=4, NUM_SMS=1, num_warps=4,
                debug_counts=dc,
                BLOCK_SIZE_M=16, BLOCK_SIZE_N=6240,
                SUB_BLOCK_SIZE_N=416, BLOCK_SIZE_K=256, NUM_STAGES=ns,
            )
            torch.npu.synchronize()
            print(f"  {name}: OK  c_sum={c.sum().item():.1f}")
        except Exception as e:
            msg = str(e)[:200]
            print(f"  {name}: {msg}")
