"""
Demo 7: 验证 for-range + atomic_add + 条件执行 —— 完整模拟动态调度
"""
import torch
import triton
import triton.language as tl


@triton.jit
def atomic_demo_for_dynamic(
    work_queue_start,   # [1,] int32
    num_tasks_ptr,      # [1,] int32
    result_ptr,         # [TOTAL_TASKS,] int32
    MAX_PER_SM: tl.constexpr,  # 每个 SM 最多抢几次
    NUM_SMS: tl.constexpr,
    TOTAL_TASKS: tl.constexpr,
):
    pid = tl.program_id(0)
    total = tl.load(num_tasks_ptr)

    for i in range(MAX_PER_SM):
        cur = tl.atomic_add(work_queue_start, 1)
        if cur < total:
            # 执行 task: 记录 SM id
            tl.store(result_ptr + cur, pid)


def test_for_dynamic(device='npu'):
    NUM_SMS = 4
    TOTAL_TASKS = 50
    MAX_PER_SM = TOTAL_TASKS  # 保证抢完

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    num_tasks = torch.tensor([TOTAL_TASKS], dtype=torch.int32, device=device)
    result = torch.full((TOTAL_TASKS,), -1, dtype=torch.int32, device=device)

    try:
        atomic_demo_for_dynamic[(NUM_SMS, 1, 1)](
            counter, num_tasks, result,
            MAX_PER_SM=MAX_PER_SM, NUM_SMS=NUM_SMS, TOTAL_TASKS=TOTAL_TASKS,
        )
        torch.npu.synchronize()
        vals = result.tolist()
        unclaimed = vals.count(-1)
        from collections import Counter
        sm_counts = Counter(v for v in vals if v >= 0)
        print(f"  counter={counter.item()}  "
              f"unclaimed={unclaimed}  "
              f"SM分布={dict(sorted(sm_counts.items()))}")
        assert unclaimed == 0, f"{unclaimed} tasks unclaimed!"
        assert len(set(vals)) == NUM_SMS + 1, f"expected {NUM_SMS} SMs + -1 sentinel"
        print("Demo 7 PASSED!")
    except Exception as e:
        print(f"Demo 7 FAILED: {str(e)[:300]}")


if __name__ == "__main__":
    test_for_dynamic()
