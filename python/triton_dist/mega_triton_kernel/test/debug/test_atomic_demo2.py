"""
atomic_add in for/while loops - Bisheng compiler crash investigation

Demo 1 已通过: atomic_add 在非 loop 场景 ✓
Demo 2 已失败: atomic_add 在 while loop 中, Bisheng SIGSEGV
现在验证:
  Demo 4: atomic_add 在 for i in range(n) 中 → 和 while 一样吗?
  Demo 5: atomic_add 在 tl.static_range 中 (编译期展开, 无 runtime loop) → 能绕过吗?
"""

import torch
import triton
import triton.language as tl


# ─── Demo 4: for i in range(n) ───
@triton.jit
def atomic_demo_for_range(
    counter_ptr,
    num_tasks_ptr,
    result_ptr,
    MAX_TASKS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    total = tl.load(num_tasks_ptr)
    write_base = result_ptr + pid * MAX_TASKS

    for i in range(MAX_TASKS):
        cur = tl.atomic_add(counter_ptr, 1)
        if cur < total:
            tl.store(write_base + i, cur)
        else:
            # 超出的不计
            pass


def test_for_range(device='npu'):
    NUM_SMS = 4
    MAX_TASKS = 8
    total_tasks = 20
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    num_tasks = torch.tensor([total_tasks], dtype=torch.int32, device=device)
    result = torch.full((NUM_SMS * MAX_TASKS,), -1, dtype=torch.int32, device=device)

    try:
        atomic_demo_for_range[(NUM_SMS, 1, 1)](counter, num_tasks, result,
                                                MAX_TASKS=MAX_TASKS, NUM_SMS=NUM_SMS)
        torch.npu.synchronize()
        vals = [v for v in result.tolist() if v >= 0]
        print(f"  counter={counter.item()}, grabbed={len(vals)}, unique={len(set(vals))}")
        print("Demo 4 PASSED")
    except Exception as e:
        print(f"Demo 4 FAILED: {str(e)[:200]}")


# ─── Demo 5: tl.static_range (compile-time unrolled) ───
# 用较小的 STATIC_N 验证编译期展开能否绕过 loop 限制
@triton.jit
def atomic_demo_static_range(
    counter_ptr,
    result_ptr,
    STATIC_N: tl.constexpr,   # 编译期常量, 每次 kernel 调用固定
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    write_base = result_ptr + pid * STATIC_N

    for i in tl.static_range(STATIC_N):
        cur = tl.atomic_add(counter_ptr, 1)
        tl.store(write_base + i, cur)


def test_static_range(device='npu'):
    NUM_SMS = 4
    STATIC_N = 8
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    result = torch.full((NUM_SMS * STATIC_N,), -1, dtype=torch.int32, device=device)

    try:
        atomic_demo_static_range[(NUM_SMS, 1, 1)](counter, result,
                                                   STATIC_N=STATIC_N, NUM_SMS=NUM_SMS)
        torch.npu.synchronize()
        vals = result.tolist()
        print(f"  counter={counter.item()}, vals={vals}, unique={len(set(vals))}")
        print("Demo 5 PASSED")
    except Exception as e:
        print(f"Demo 5 FAILED: {str(e)[:200]}")


# ─── Demo 6: while + barrier (不用 atomic_add 的 while 能编译吗?) ───
@triton.jit
def demo_while_no_atomic(
    counter_ptr,
    num_tasks_ptr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    total = tl.load(num_tasks_ptr)

    cur = pid
    while cur < total:
        cur += NUM_SMS

    # just verify while itself doesn't crash
    tl.store(counter_ptr, cur)


def test_while_no_atomic(device='npu'):
    NUM_SMS = 4
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    num_tasks = torch.tensor([100], dtype=torch.int32, device=device)
    try:
        demo_while_no_atomic[(NUM_SMS, 1, 1)](counter, num_tasks, NUM_SMS=NUM_SMS)
        torch.npu.synchronize()
        print(f"  counter={counter.item()}")
        print("Demo 6 PASSED")
    except Exception as e:
        print(f"Demo 6 FAILED: {str(e)[:200]}")


if __name__ == "__main__":
    import sys
    device = 'npu' if len(sys.argv) < 2 else sys.argv[1]
    print(f"=== atomic_add loop investigation on device={device} ===")
    test_for_range(device)
    test_static_range(device)
    test_while_no_atomic(device)
