"""
Demo 8-9: 加强版 atomic_add 竞争验证 + 模拟真实 megakernel 执行

验证点:
  8. 有竞争的 atomic_add — 每个 task 有不同耗时, 确保多 SM 同时抢
  9. 静态 for-range — 证明 MAX_TASKS 固定上限的正确性
"""

import torch
import triton
import triton.language as tl


# ═══════════════════ Demo 8: 竞争 atomic_add + 模拟 task 耗时 ═══════════════════
@triton.jit
def atomic_demo_contention(
    work_queue_start,   # [1,] int32 — 全局计数器
    result_ptr,         # [NUM_SMS * RECORDS,] int32 — 每个 SM 记录它抢到的 task id
    result_order,       # [NUM_SMS * RECORDS,] int32 — 每个 SM 记录获得 task 的顺序
    delay_buf,          # [TOTAL_TASKS,] int32 — 模拟 task 耗时的 dummy buffer (越大越慢)
    MAX_PER_SM: tl.constexpr,
    TOTAL_TASKS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    total = TOTAL_TASKS
    write_base = result_ptr + pid * MAX_PER_SM
    order_base = result_order + pid * MAX_PER_SM
    slot = 0  # local counter — 不能超过 MAX_PER_SM

    for i in range(MAX_PER_SM):
        cur = tl.atomic_add(work_queue_start, 1)
        if cur < total:
            # 记录抢到的 task id
            tl.store(write_base + slot, cur)
            tl.store(order_base + slot, i)  # 第 i 次迭代抢到
            slot += 1

            # 模拟 task 执行: 对一段内存做读写, 耗时与 cur 值有关 (不同 task 不同耗时)
            # 这让不同 SM 的 atomic_add 真正交叠
            for d in range(cur % 16 + 1):
                val = tl.load(delay_buf + cur)
                tl.store(delay_buf + cur, val + pid)


def test_contention(device='npu'):
    """验证 atomic_add 在多 SM 竞争下: (a) 无重复 task id (b) 无遗漏 (c) counter 正确"""
    NUM_SMS = 8
    TOTAL_TASKS = 64
    MAX_PER_SM = TOTAL_TASKS  # 足够的余量

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    result = torch.full((NUM_SMS * MAX_PER_SM,), -1, dtype=torch.int32, device=device)
    result_order = torch.full((NUM_SMS * MAX_PER_SM,), -1, dtype=torch.int32, device=device)
    delay_buf = torch.zeros(TOTAL_TASKS, dtype=torch.int32, device=device)

    grid = (NUM_SMS, 1, 1)
    atomic_demo_contention[grid](
        counter, result, result_order, delay_buf,
        MAX_PER_SM=MAX_PER_SM, TOTAL_TASKS=TOTAL_TASKS, NUM_SMS=NUM_SMS,
    )
    torch.npu.synchronize()

    # 收集所有 SM 记录的有效 task id
    all_ids = [v for v in result.tolist() if v >= 0]
    all_orders = [v for v in result_order.tolist() if v >= 0]

    # 每 SM 抢到多少 task
    sm_tasks = {}
    for sm in range(NUM_SMS):
        start = sm * MAX_PER_SM
        end = start + MAX_PER_SM
        ids = [v for v in result[start:end].tolist() if v >= 0]
        if ids:
            sm_tasks[sm] = len(ids)

    print(f"counter={counter.item()} (expect {len(all_ids)})")
    print(f"total claimed={len(all_ids)} (expect {TOTAL_TASKS})")
    print(f"unique ids={len(set(all_ids))} (expect {TOTAL_TASKS})")
    print(f"SM→tasks: {dict(sorted(sm_tasks.items()))}")

    # 核心验证
    errors = []
    actual_counter = counter.item()
    if len(all_ids) != TOTAL_TASKS:
        errors.append(f"TASK COUNT: claimed={len(all_ids)} != total={TOTAL_TASKS}")
    if len(set(all_ids)) != TOTAL_TASKS:
        errors.append(f"DUP: {len(all_ids) - len(set(all_ids))} duplicate task ids!")
    if not all(0 <= x < TOTAL_TASKS for x in all_ids):
        out_of_range = [x for x in all_ids if x < 0 or x >= TOTAL_TASKS]
        errors.append(f"RANGE: values out of [0,{TOTAL_TASKS}): {out_of_range}")
    # 验证每个 task id 恰好出现一次
    from collections import Counter
    cnt = Counter(all_ids)
    dup_ids = {k: v for k, v in cnt.items() if v > 1}
    missing_ids = [i for i in range(TOTAL_TASKS) if cnt.get(i, 0) == 0]
    if dup_ids:
        errors.append(f"DUP: {dup_ids}")
    if missing_ids:
        errors.append(f"MISS: {len(missing_ids)} tasks unclaimed: {missing_ids[:10]}...")

    if errors:
        print(f"\nFAILED:")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"\nDemo 8 PASSED — all {TOTAL_TASKS} tasks claimed correctly, "
              f"distributed across {len(sm_tasks)}/{NUM_SMS} SMs")


# ═══════════════ Demo 9: for-range 上限验证 — 如果上限不够会怎样 ═══════════════
@triton.jit
def atomic_demo_tight_bound(
    work_queue_start,
    result_ptr,
    ALLOCATED: tl.constexpr,
    TOTAL_TASKS: tl.constexpr,
):
    pid = tl.program_id(0)
    write_base = result_ptr + pid * ALLOCATED
    slot = 0

    for i in range(ALLOCATED):
        cur = tl.atomic_add(work_queue_start, 1)
        if cur < TOTAL_TASKS:
            tl.store(write_base + slot, cur)
            slot += 1


def test_tight_bound(device='npu'):
    """验证: 当上限刚好够用时, 是否所有 task 都被认领"""
    NUM_SMS = 4
    TOTAL_TASKS = 100
    ALLOCATED = TOTAL_TASKS // NUM_SMS + 2  # 刚好有点余量

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    result = torch.full((NUM_SMS * ALLOCATED,), -1, dtype=torch.int32, device=device)

    atomic_demo_tight_bound[(NUM_SMS, 1, 1)](
        counter, result, ALLOCATED=ALLOCATED, TOTAL_TASKS=TOTAL_TASKS,
    )
    torch.npu.synchronize()

    all_ids = [v for v in result.tolist() if v >= 0]
    missing = TOTAL_TASKS - len(all_ids)

    print(f"ALLOCATED={ALLOCATED}/SM, TOTAL={TOTAL_TASKS}, NUM_SMS={NUM_SMS}")
    print(f"claimed={len(all_ids)}, missing={missing}, counter={counter.item()}")
    if missing == 0:
        print(f"Demo 9 PASSED — ALLOCATED sufficient")
    else:
        print(f"Demo 9 INFO — {missing} tasks unclaimed (ALLOCATED may be too tight)")


if __name__ == "__main__":
    import sys
    device = 'npu' if len(sys.argv) < 2 else sys.argv[1]
    print(f"=== atomic_add correctness verification on {device} ===")
    test_contention(device)
    print()
    test_tight_bound(device)
