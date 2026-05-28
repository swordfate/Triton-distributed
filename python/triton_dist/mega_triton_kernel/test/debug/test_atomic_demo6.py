"""
Demo 12: 模拟真实 megakernel 动态调度模式 — 排查卡死

逐步放大 MAX_TASKS, 看 Bisheng 编译和运行是否正确:
  12a: MAX_TASKS=32   (小)
  12b: MAX_TASKS=128  (中)
  12c: MAX_TASKS=512  (中)
  12d: MAX_TASKS=2048 (大)
  12e: MAX_TASKS=8192 (超大 — megakernel 级别)
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def atomic_for_range_scale_test(
    work_queue_start,
    num_tasks_ptr,
    result_sm_id,       # [TOTAL_TASKS,] int32
    MAX_TASKS: tl.constexpr,
    TOTAL_TASKS: tl.constexpr,
):
    pid = tl.program_id(0)
    total = tl.load(num_tasks_ptr)

    for i in range(MAX_TASKS):
        cur = tl.atomic_add(work_queue_start, 1)
        if cur < total:
            tl.store(result_sm_id + cur, pid)


def test_scale(device='npu'):
    """逐步放大, 每一步都验证编译+运行+数据正确性"""
    NUM_SMS = 4
    scales = [32, 128, 512, 2048, 4096]

    for total in scales:
        counter = torch.zeros(1, dtype=torch.int32, device=device)
        num_tasks = torch.tensor([total], dtype=torch.int32, device=device)
        result = torch.full((total,), -1, dtype=torch.int32, device=device)

        t0 = time.perf_counter()
        try:
            atomic_for_range_scale_test[(NUM_SMS, 1, 1)](
                counter, num_tasks, result,
                MAX_TASKS=total, TOTAL_TASKS=total,
            )
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000

            sm_ids = result.tolist()
            missing = sm_ids.count(-1)
            dup = len(sm_ids) - len(set(sm_ids)) + 1  # -1 不算重复
            print(f"TOTAL={total:5d}  elapsed={elapsed:7.2f}ms  "
                  f"counter={counter.item():5d}  missing={missing}  dup={dup}  "
                  f"OK" if missing == 0 else f"MISSING={missing}!")
        except Exception as e:
            msg = str(e)
            if "SIGSEGV" in msg or "Segmentation" in msg:
                print(f"TOTAL={total:5d}  SIGSEGV — Bisheng crash at this scale!")
            elif "MLIR" in msg:
                print(f"TOTAL={total:5d}  MLIR error: {msg[:100]}")
            else:
                print(f"TOTAL={total:5d}  ERROR: {msg[:100]}")


if __name__ == "__main__":
    import sys
    device = 'npu' if len(sys.argv) < 2 else sys.argv[1]
    print(f"=== Scale test on {device} ===")
    test_scale(device)
