"""
Demo 13: atomic_add 在循环外 (一次抢 chunk), 循环内无原子操作
         —— 彻底避免 atomic_add + loop 的组合

Demo 14: chunk 不够用时多次抢 chunk (但每次都在独立代码块, 不在 loop 内)
"""

import torch
import triton
import triton.language as tl
import time


# ═══ Demo 13: 一次 atomic_add 抢 chunk, 循环内无原子操作 ═══
@triton.jit
def atomic_demo_chunk_once(
    work_queue_start,   # [1,] int32
    result_sm_id,       # [TOTAL_TASKS,] int32
    scratch_buf,
    CHUNK_SIZE: tl.constexpr,
    TOTAL_TASKS: tl.constexpr,
    HEAVY_ITERS: tl.constexpr,
    HEAVY_DIM: tl.constexpr,
):
    pid = tl.program_id(0)

    # ─── 原子操作在循环外, 只调一次 ───
    my_start = tl.atomic_add(work_queue_start, CHUNK_SIZE)
    my_end = my_start + CHUNK_SIZE
    if my_end > TOTAL_TASKS:
        my_end = TOTAL_TASKS

    # ─── 循环内无原子操作 ───
    for cur in range(my_start, my_end):
        # 记录
        tl.store(result_sm_id + cur, pid)

        # 模拟计算
        offs = tl.arange(0, HEAVY_DIM)
        row_base = scratch_buf + cur * HEAVY_DIM
        acc = tl.zeros([HEAVY_DIM], dtype=tl.float32)
        for _ in range(HEAVY_ITERS):
            val = tl.load(row_base + offs)
            acc = acc + val * 0.99
            tl.store(row_base + offs, acc)


def test_chunk_once(device='npu'):
    TOTAL_TASKS = 128
    NUM_SMS = 8
    CHUNK_SIZE = TOTAL_TASKS // NUM_SMS + 1
    HEAVY_ITERS = 200
    HEAVY_DIM = 512

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    result = torch.full((TOTAL_TASKS,), -1, dtype=torch.int32, device=device)
    scratch = torch.randn(TOTAL_TASKS, HEAVY_DIM, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    atomic_demo_chunk_once[(NUM_SMS, 1, 1)](
        counter, result, scratch,
        CHUNK_SIZE=CHUNK_SIZE, TOTAL_TASKS=TOTAL_TASKS,
        HEAVY_ITERS=HEAVY_ITERS, HEAVY_DIM=HEAVY_DIM,
    )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t0

    sm_ids = result.tolist()
    from collections import Counter
    sm_counts = Counter(sm_ids)
    n_sms = len([k for k, v in sm_counts.items() if v > 0])
    missing = sm_ids.count(-1)
    dup = len(sm_ids) - len(set(sm_ids)) + (1 if -1 in sm_ids else 0)

    print(f"chunk_once: elapsed={elapsed*1000:.0f}ms  "
          f"counter={counter.item()}  missing={missing}  "
          f"SMs={n_sms}/{NUM_SMS}  SM→count={dict(sorted(sm_counts.items()))}")
    if missing == 0 and dup <= 0:
        print("  PASSED")
    else:
        print(f"  FAIL: missing={missing} dup={dup}")


# ═══ Demo 14: 多次抢 chunk (展开, 不在 loop 内) ═══
# 固定展开 4 个 chunk, 足够覆盖
@triton.jit
def atomic_demo_chunk_unrolled(
    work_queue_start,
    result_sm_id,
    scratch_buf,
    CHUNK_SIZE: tl.constexpr,
    TOTAL_TASKS: tl.constexpr,
    HEAVY_ITERS: tl.constexpr,
    HEAVY_DIM: tl.constexpr,
):
    pid = tl.program_id(0)

    # ─── 4 个 chunk, 不在任何 loop 内 ───
    # Chunk 1
    my_start1 = tl.atomic_add(work_queue_start, CHUNK_SIZE)
    if my_start1 < TOTAL_TASKS:
        my_end1 = my_start1 + CHUNK_SIZE
        if my_end1 > TOTAL_TASKS:
            my_end1 = TOTAL_TASKS
        for cur in range(my_start1, my_end1):
            tl.store(result_sm_id + cur, pid)
            offs = tl.arange(0, HEAVY_DIM)
            row_base = scratch_buf + cur * HEAVY_DIM
            acc = tl.zeros([HEAVY_DIM], dtype=tl.float32)
            for _ in range(HEAVY_ITERS):
                val = tl.load(row_base + offs)
                acc = acc + val * 0.99
                tl.store(row_base + offs, acc)

    # Chunk 2
    my_start2 = tl.atomic_add(work_queue_start, CHUNK_SIZE)
    if my_start2 < TOTAL_TASKS:
        my_end2 = my_start2 + CHUNK_SIZE
        if my_end2 > TOTAL_TASKS:
            my_end2 = TOTAL_TASKS
        for cur in range(my_start2, my_end2):
            tl.store(result_sm_id + cur, pid)
            offs = tl.arange(0, HEAVY_DIM)
            row_base = scratch_buf + cur * HEAVY_DIM
            acc = tl.zeros([HEAVY_DIM], dtype=tl.float32)
            for _ in range(HEAVY_ITERS):
                val = tl.load(row_base + offs)
                acc = acc + val * 0.99
                tl.store(row_base + offs, acc)

    # Chunk 3
    my_start3 = tl.atomic_add(work_queue_start, CHUNK_SIZE)
    if my_start3 < TOTAL_TASKS:
        my_end3 = my_start3 + CHUNK_SIZE
        if my_end3 > TOTAL_TASKS:
            my_end3 = TOTAL_TASKS
        for cur in range(my_start3, my_end3):
            tl.store(result_sm_id + cur, pid)
            offs = tl.arange(0, HEAVY_DIM)
            row_base = scratch_buf + cur * HEAVY_DIM
            acc = tl.zeros([HEAVY_DIM], dtype=tl.float32)
            for _ in range(HEAVY_ITERS):
                val = tl.load(row_base + offs)
                acc = acc + val * 0.99
                tl.store(row_base + offs, acc)

    # Chunk 4
    my_start4 = tl.atomic_add(work_queue_start, CHUNK_SIZE)
    if my_start4 < TOTAL_TASKS:
        my_end4 = my_start4 + CHUNK_SIZE
        if my_end4 > TOTAL_TASKS:
            my_end4 = TOTAL_TASKS
        for cur in range(my_start4, my_end4):
            tl.store(result_sm_id + cur, pid)
            offs = tl.arange(0, HEAVY_DIM)
            row_base = scratch_buf + cur * HEAVY_DIM
            acc = tl.zeros([HEAVY_DIM], dtype=tl.float32)
            for _ in range(HEAVY_ITERS):
                val = tl.load(row_base + offs)
                acc = acc + val * 0.99
                tl.store(row_base + offs, acc)


def test_chunk_unrolled(device='npu'):
    TOTAL_TASKS = 128
    NUM_SMS = 8
    CHUNK_SIZE = TOTAL_TASKS // NUM_SMS
    HEAVY_ITERS = 50
    HEAVY_DIM = 256

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    result = torch.full((TOTAL_TASKS,), -1, dtype=torch.int32, device=device)
    scratch = torch.randn(TOTAL_TASKS, HEAVY_DIM, dtype=torch.float32, device=device)

    try:
        atomic_demo_chunk_unrolled[(NUM_SMS, 1, 1)](
            counter, result, scratch,
            CHUNK_SIZE=CHUNK_SIZE, TOTAL_TASKS=TOTAL_TASKS,
            HEAVY_ITERS=HEAVY_ITERS, HEAVY_DIM=HEAVY_DIM,
        )
        torch.npu.synchronize()
        sm_ids = result.tolist()
        from collections import Counter
        sm_counts = Counter(sm_ids)
        missing = sm_ids.count(-1)
        print(f"chunk_unrolled: counter={counter.item()}  missing={missing}  "
              f"SM→count={dict(sorted(sm_counts.items()))}")
        if missing == 0:
            print("  PASSED")
        else:
            print(f"  missing {missing} tasks")
    except Exception as e:
        print(f"chunk_unrolled FAILED: {str(e)[:200]}")


if __name__ == "__main__":
    import sys
    device = 'npu' if len(sys.argv) < 2 else sys.argv[1]
    print(f"=== chunk scheduling on {device} ===")
    test_chunk_once(device)
    print()
    test_chunk_unrolled(device)
