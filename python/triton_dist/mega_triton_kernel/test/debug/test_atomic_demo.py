"""
atomic_add 动态任务分配 demo (Ascend 910B3)

模拟: 20 个 AICore 竞争一个全局计数器, 用 atomic_add 的返回值
作为唯一 task id, 各自从全局队列中取 task 并"执行"。

目的: 在独立最小 Triton kernel 中验证 atomic_add 在 while loop 中的行为,
      排除 megakernel 其他部分干扰。
"""

import torch
import triton
import triton.language as tl


# ─── Demo 1: 最简 atomic_add, 非 loop 场景 ───
@triton.jit
def atomic_demo_simple(
    counter_ptr,       # [1,] int32 — 全局计数器
    result_ptr,        # [NUM_SMS,] int32 — 每个 block 记录自己抢到的值
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    # 最简单的调用: 无 sem/scope 参数 (用默认 acq_rel + gpu)
    my_val = tl.atomic_add(counter_ptr, 1)
    tl.store(result_ptr + pid, my_val)


def test_simple(device='npu'):
    NUM_SMS = 4  # 用 4 个 block 测试
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    result = torch.full((NUM_SMS,), -1, dtype=torch.int32, device=device)

    atomic_demo_simple[(NUM_SMS, 1, 1)](counter, result, NUM_SMS=NUM_SMS)
    torch.npu.synchronize()

    print(f"counter = {counter.item()}")
    print(f"results = {result.tolist()}")
    # 期望: results 包含 0,1,2,3 (顺序不保证), counter=4
    assert counter.item() == NUM_SMS, f"expected {NUM_SMS}, got {counter.item()}"
    assert sorted(result.tolist()) == list(range(NUM_SMS)), f"unexpected results {result.tolist()}"
    print("Demo 1 PASSED")


# ─── Demo 2: atomic_add 在 while loop 中 ───
@triton.jit
def atomic_demo_while_loop(
    work_queue_start,  # [1,] int32 — 全局原子计数器
    num_tasks,         # [1,] int32 — 总 task 数
    result_ptr,        # [NUM_SMS * MAX_TASKS,] int32 — 每个 block 记录抢到的 task id
    NUM_SMS: tl.constexpr,
    MAX_TASKS: tl.constexpr,  # 每个 block 最多记录的 task 数
):
    pid = tl.program_id(0)
    total = tl.load(num_tasks)

    # 每个 block 的记录指针
    write_base = result_ptr + pid * MAX_TASKS
    write_idx = 0

    # 原子抢第一个 task
    cur_idx = tl.atomic_add(work_queue_start, 1)

    while cur_idx < total:
        # "执行" task: 记录到 result
        if write_idx < MAX_TASKS:
            tl.store(write_base + write_idx, cur_idx)

        # 抢下一个 task
        write_idx += 1
        cur_idx = tl.atomic_add(work_queue_start, 1)


def test_while_loop(device='npu'):
    NUM_SMS = 4
    MAX_TASKS = 32  # 每个 block 多抢几个, 直到耗尽
    total_tasks = 50

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    num_tasks = torch.tensor([total_tasks], dtype=torch.int32, device=device)
    result = torch.full((NUM_SMS * MAX_TASKS,), -1, dtype=torch.int32, device=device)

    atomic_demo_while_loop[(NUM_SMS, 1, 1)](
        counter, num_tasks, result,
        NUM_SMS=NUM_SMS, MAX_TASKS=MAX_TASKS,
    )
    torch.npu.synchronize()

    print(f"counter = {counter.item()} (expect {total_tasks})")
    # 收集所有 block 抢到的 task id
    all_vals = result.tolist()
    all_vals = [v for v in all_vals if v >= 0]
    print(f"  total grabbed = {len(all_vals)}, unique = {len(set(all_vals))}, "
          f"min = {min(all_vals)}, max = {max(all_vals)}")

    assert len(all_vals) == total_tasks, f"expected {total_tasks} tasks, got {len(all_vals)}"
    assert len(set(all_vals)) == total_tasks, "duplicate task ids!"
    print("Demo 2 PASSED")


# ─── Demo 3: 模拟真实动态调度 —— while loop + 条件退出 ───
@triton.jit
def atomic_demo_realistic(
    work_queue_start,  # [1,] int32 — 全局原子计数器
    num_tasks_ptr,     # [1,] int32 — 总 task 数
    result_ptr,        # [MAX_RECORDS,] int32 — 所有 block 的记录 (按 task idx 索引)
    MAX_RECORDS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    total = tl.load(num_tasks_ptr)
    record_base = result_ptr

    # 抢第一个 task
    cur_idx = tl.atomic_add(work_queue_start, 1)
    if cur_idx >= total:
        return

    # while loop + 提前退出
    while cur_idx < total:
        # "执行" task: 写入记录 (task_idx → pid)
        tl.store(record_base + cur_idx, pid)

        # 抢下一个
        cur_idx = tl.atomic_add(work_queue_start, 1)


def test_realistic(device='npu'):
    NUM_SMS = 4
    total_tasks = 40  # 模拟 40 个 task, 4 个 SM 竞争

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    num_tasks = torch.tensor([total_tasks], dtype=torch.int32, device=device)
    result = torch.full((total_tasks,), -1, dtype=torch.int32, device=device)

    atomic_demo_realistic[(NUM_SMS, 1, 1)](
        counter, num_tasks, result,
        MAX_RECORDS=total_tasks, NUM_SMS=NUM_SMS,
    )
    torch.npu.synchronize()

    results = result.tolist()
    print(f"counter = {counter.item()} (expect {total_tasks})")
    print(f"  task→SM 映射 (前20): {results[:20]}")

    # 验证: 每个 task 恰好被一个 SM 认领
    assert counter.item() == total_tasks, f"expected counter {total_tasks}, got {counter.item()}"
    sm_ids = set(results)
    assert -1 not in sm_ids, "some tasks not claimed!"
    assert len(sm_ids) >= 2, f"all tasks claimed by one SM? sm_ids={sm_ids}"

    # 统计各 SM 认领的 task 数
    from collections import Counter
    sm_counts = Counter(results)
    print(f"  SM→task_count: {dict(sorted(sm_counts.items()))}")

    print("Demo 3 PASSED")


if __name__ == "__main__":
    import sys
    device = 'npu' if len(sys.argv) < 2 else sys.argv[1]
    print(f"Running atomic demos on device={device}")
    print("=" * 60)

    try:
        test_simple(device)
    except Exception as e:
        print(f"Demo 1 FAILED: {e}")

    try:
        test_while_loop(device)
    except Exception as e:
        print(f"Demo 2 FAILED: {e}")

    try:
        test_realistic(device)
    except Exception as e:
        print(f"Demo 3 FAILED: {e}")
