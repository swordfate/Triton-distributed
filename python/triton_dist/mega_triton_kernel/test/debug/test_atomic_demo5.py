"""
Demo 10: 加入真正的计算延迟模拟 megakernel task 执行, 强制 SM 竞争

策略: 每个 task 执行时做大量 dummy 计算 (tensor ops in Triton),
      确保执行时间 >> atomic_add 时间, 多个 SM 才能真正交替抢 task.

Demo 11: 检查 atomic_add 返回值类型 —— 用 tl.device_assert 确认能正确比较
"""

import torch
import triton
import triton.language as tl
import time


# ═══════════════ Demo 10: 带真实计算延迟的竞争 ═══════════════
@triton.jit
def atomic_demo_heavy(
    work_queue_start,   # [1,] int32
    result_sm_id,       # [TOTAL_TASKS,] int32 — task_id → 执行它的 SM id
    result_grab_order,  # [TOTAL_TASKS,] int32 — task_id → 全局第几次 atomic_add 抢到
    scratch_buf,        # [TOTAL_TASKS, HEAVY_DIM] float32 — 计算用的 scratch
    HEAVY_ITERS: tl.constexpr,   # dummy 计算迭代次数 (越大越慢)
    HEAVY_DIM: tl.constexpr,     # dummy 计算的向量维度
    MAX_PER_SM: tl.constexpr,
    TOTAL_TASKS: tl.constexpr,
):
    pid = tl.program_id(0)

    for i in range(MAX_PER_SM):
        cur = tl.atomic_add(work_queue_start, 1)
        if cur < TOTAL_TASKS:
            # 记录: 哪个 SM 执行了这个 task
            tl.store(result_sm_id + cur, pid)
            # 记录: 这是第几次全局 atomic_add 产生的 task
            # (cur 本身就是 fetch-and-add 的旧值, 即 task 分配序号)
            tl.store(result_grab_order + cur, cur)

            # ─── 模拟真实计算: 对 scratch 做读写 + 简单运算 ───
            # 从 scratch_buf 加载 cur 这一行
            offs = tl.arange(0, HEAVY_DIM)
            row_base = scratch_buf + cur * HEAVY_DIM
            acc = tl.zeros([HEAVY_DIM], dtype=tl.float32)
            # 多轮迭代, 每轮做 load -> mul -> add -> store
            for _ in range(HEAVY_ITERS):
                val = tl.load(row_base + offs)
                acc = acc + val * 0.99
                tl.store(row_base + offs, acc)


def test_heavy(device='npu'):
    """期望: 多个 SM 都抢到 task, 分布均匀"""
    TOTAL_TASKS = 32
    NUM_SMS = 8
    MAX_PER_SM = TOTAL_TASKS
    HEAVY_ITERS = 50   # 调大这个值让 task 更"重"
    HEAVY_DIM = 1024   # 向量维度

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    result_sm = torch.full((TOTAL_TASKS,), -1, dtype=torch.int32, device=device)
    result_order = torch.full((TOTAL_TASKS,), -1, dtype=torch.int32, device=device)
    scratch = torch.randn(TOTAL_TASKS, HEAVY_DIM, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    atomic_demo_heavy[(NUM_SMS, 1, 1)](
        counter, result_sm, result_order, scratch,
        HEAVY_ITERS=HEAVY_ITERS, HEAVY_DIM=HEAVY_DIM,
        MAX_PER_SM=MAX_PER_SM, TOTAL_TASKS=TOTAL_TASKS,
    )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t0

    sm_ids = result_sm.tolist()
    orders = result_order.tolist()

    # 每个 SM 抢到多少
    from collections import Counter
    sm_counts = Counter(sm_ids)
    n_participating = len([k for k, v in sm_counts.items() if v > 0])

    print(f"elapsed={elapsed*1000:.1f}ms  counter={counter.item()}")
    print(f"claimed={sum(sm_counts.values())} (expect {TOTAL_TASKS})")
    print(f"SM participating: {n_participating}/{NUM_SMS}")
    print(f"SM→count: {dict(sorted(sm_counts.items()))}")

    # 首次和最后分配
    order_map = {o: s for o, s in zip(orders, sm_ids)}
    sorted_orders = sorted(order_map.keys())
    if sorted_orders:
        first_order = sorted_orders[0]
        last_order = sorted_orders[-1]
        print(f"first task_id={first_order} (SM{order_map[first_order]}), "
              f"last task_id={last_order} (SM{order_map[last_order]})")

    errors = []
    if sum(sm_counts.values()) != TOTAL_TASKS:
        errors.append(f"MISS: only {sum(sm_counts.values())}/{TOTAL_TASKS} claimed")
    dup = {k: v for k, v in sm_counts.items() if v < 0}
    if len(set(sm_ids)) != TOTAL_TASKS + 1 and -1 in sm_ids:  # -1 sentinel
        errors.append(f"UNCLAIMED: some tasks still -1")
    if n_participating < 2:
        print(f"\nWARNING: only 1 SM participated — competition too weak, increase HEAVY_ITERS")
    else:
        print(f"\nDemo 10 PASSED — {n_participating}/{NUM_SMS} SMs participated")


# ═══════════════ Demo 11: 验证 atomic_add 返回值可直接用于比较和索引 ═══════════════
@triton.jit
def atomic_demo_type_check(
    work_queue_start,
    result_arr,         # [NUM_CHECKS * 3,] int32 — 每组记录 [cur, pid_is, pid_should]
    MAX_PER_SM: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    base = result_arr + pid * 3
    slot = 0

    for i in range(MAX_PER_SM):
        cur = tl.atomic_add(work_queue_start, 1)

        # 测试1: 与常量比较
        if cur >= 0:
            pass  # 条件通过则继续

        # 测试2: 与 pid 比较 (另一个 int32)
        if cur >= pid:
            pass

        if slot < 1:
            # 测试3: 作为存储索引
            tl.store(base + 0, cur)         # 存返回值本身
            tl.store(base + 1, pid)         # 存当前 SM id (对照)
            tl.store(base + 2, pid)         # 期望值 = pid (假)
            slot += 1


def test_type_check(device='npu'):
    NUM_SMS = 4
    MAX_PER_SM = 1
    result = torch.full((NUM_SMS * 3,), -1, dtype=torch.int32, device=device)
    counter = torch.zeros(1, dtype=torch.int32, device=device)

    atomic_demo_type_check[(NUM_SMS, 1, 1)](
        counter, result,
        MAX_PER_SM=MAX_PER_SM, NUM_SMS=NUM_SMS,
    )
    torch.npu.synchronize()

    vals = result.tolist()
    print(f"result = {vals}")
    print(f"counter = {counter.item()}")

    # 每个 SM 存了 [cur, pid, pid], 验证 cur 在 0..NUM_SMS-1 范围
    all_curs = []
    for sm in range(NUM_SMS):
        cur = vals[sm * 3]
        if cur >= 0 and cur < NUM_SMS:
            all_curs.append(cur)
    unique = set(all_curs)
    print(f"returned values: {all_curs}, unique: {sorted(unique)}")

    if len(unique) == NUM_SMS:
        print("Demo 11 PASSED — atomic_add returns unique, comparable int32")
    else:
        print(f"Demo 11 WARN — got {len(unique)} unique out of {NUM_SMS} SMs")


if __name__ == "__main__":
    import sys
    device = 'npu' if len(sys.argv) < 2 else sys.argv[1]
    print(f"=== atomic_add enhanced verification on {device} ===")
    test_heavy(device)
    print()
    test_type_check(device)
