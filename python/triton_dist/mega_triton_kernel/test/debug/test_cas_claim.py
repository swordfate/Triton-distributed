"""
CAS 动态认领 Demo: atomic_add 探针 + atomic_cas 不同地址认领

核心思路:
  - atomic_add 推进全局探针
  - 不同 SM 加上不同的 pid*STAGGER 偏移 → CAS 到不同地址 → 无竞争
  - 首轮即使 atomic_add 返回值有锁步问题，不同 SM 的 CAS 地址依然不同
  - 首轮 task 执行时间不同 → 锁步打破 → 后续轮次返回值也正确

Test 1: CAS 同一地址计数器（预期有重复——验证 CAS 和 ADD 同路径）
Test 2: CAS 不同地址 scatter（官方模式，预期通过）
Test 3: 探针 + CAS 不同地址认领（主方案，预期无重复无遗漏）
"""

import torch
import triton
import triton.language as tl
import torch_npu
import time
from collections import Counter


# ═══════════════ Test 1: CAS 同一地址计数器 ═══════════════
@triton.jit
def cas_same_addr_counter(
    counter_ptr, result_ptr,
    NUM_SMS: tl.constexpr, ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    write_base = result_ptr + pid * ITERS
    slot = 0

    for i in tl.static_range(ITERS):
        old = tl.load(counter_ptr)
        result = tl.atomic_cas(counter_ptr, old, old + 1)
        tl.store(write_base + slot, result)
        slot += 1


# ═══════════════ Test 2: CAS 不同地址 scatter ═══════════════
@triton.jit
def cas_scatter(
    task_status, result_ptr,
    NUM_SMS: tl.constexpr, TOTAL: tl.constexpr,
    MAX_PER_SM: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = 0

    for i in range(MAX_PER_SM):
        cand = pid + i * NUM_SMS
        if cand < TOTAL:
            old = tl.atomic_cas(task_status + cand, 0, pid + 1)
            if old == 0:
                tl.store(result_ptr + pid * MAX_PER_SM + slot, cand)
                slot += 1


# ═══════════════ Test 3: 探针 + CAS 不同地址认领 ═══════════════
#
# 关键约束:
#   1. TOTAL 必须是 2 的幂 → 用 & (TOTAL-1) 代替 % TOTAL 避免模运算卡编译
#   2. STAGGER >= PROBE → 确保首轮不同 SM 的地址区间不重叠
#   3. 只用 tl.static_range（编译期展开），不用 for-range/while
#
@triton.jit
def probe_cas_claim(
    task_owner,        # [TOTAL] int32: 0=free, pid+1=claimed
    counter_ptr,       # [1] int32: 全局探针
    result_ptr,        # [NUM_SMS * MAX_CLAIMED] int32
    TOTAL: tl.constexpr,        # 必须 2 的幂
    NUM_SMS: tl.constexpr,
    PROBE: tl.constexpr,        # 每轮探针窗口大小
    STAGGER: tl.constexpr,      # 每 SM 地址偏移 (>= PROBE)
    ROUNDS: tl.constexpr,
    MAX_CLAIMED: tl.constexpr,
):
    pid = tl.program_id(0)
    write_base = result_ptr + pid * MAX_CLAIMED
    mask = TOTAL - 1
    slot = 0

    for r in tl.static_range(ROUNDS):
        # ① 原子推进全局探针
        base = tl.atomic_add(counter_ptr, PROBE)
        # ② 加上 pid 偏移，确保不同 SM CAS 不同地址
        base = base + pid * STAGGER

        # ③ 在探针窗口内尝试 CAS 认领（用 AND mask 代替 modulo）
        for i in tl.static_range(PROBE):
            cand = (base + i) & mask
            old = tl.atomic_cas(task_owner + cand, 0, pid + 1)

            if old == 0:
                if slot < MAX_CLAIMED:
                    tl.store(write_base + slot, cand)
                slot += 1


# ════════════════════ 验证逻辑 ════════════════════

def check_results(task_status, result_ptr, MAX_PER_SM, TOTAL, NUM_SMS, test_name):
    """
    三重验证:
      ① task_status: 每个 slot 恰好被一个 SM 认领（无遗漏）
      ② result_ptr:   各 SM 记录中，每个 task ID 只出现一次（无重复认领）
      ③ 交叉验证:     result_ptr 和 task_status 一致
    """
    status = task_status.cpu().tolist()
    records = result_ptr.cpu().tolist()
    errors = []

    # ── ① task_status 全覆盖 ──
    for i in range(TOTAL):
        v = status[i]
        if v == 0:
            errors.append(f"  task[{i}]: UNCLAIMED")
        elif v < 1 or v > NUM_SMS:
            errors.append(f"  task[{i}]: invalid {v}")

    # ── ② result_ptr 去重 ──
    all_claimed = []
    seen_ids = set()
    dup_ids = []
    for sm in range(NUM_SMS):
        for j in range(MAX_PER_SM):
            val = records[sm * MAX_PER_SM + j]
            if val >= 0:
                all_claimed.append((val, sm))
                if val in seen_ids:
                    dup_ids.append(val)
                seen_ids.add(val)

    # ── ③ 交叉验证 ──
    for task_id, sm in all_claimed:
        if task_id < TOTAL and status[task_id] != sm + 1:
            errors.append(f"  task[{task_id}]: SM{sm} says claimed "
                          f"but status={status[task_id]}")

    if dup_ids:
        errors.append(f"  DUPLICATE: {dup_ids[:20]} "
                      f"(total {len(dup_ids)})")

    n_claimed = len(all_claimed)
    missing_slots = [i for i in range(TOTAL) if status[i] == 0]

    if errors or missing_slots:
        for e in errors[:15]:
            print(e)
        if not errors and missing_slots:
            print(f"  MISSING: {len(missing_slots)} slots unclaimed")
        print(f"  total claimed={n_claimed}, unique={len(seen_ids)}")
        print(f"  {test_name}: ❌ FAILED")
        return False
    else:
        sm_counts = Counter(v for v in status if v > 0)
        n_sms = len(sm_counts)
        print(f"  {TOTAL} tasks by {n_sms}/{NUM_SMS} SMs: "
              f"{dict(sorted(sm_counts.items()))}")
        print(f"  cross-check: all records consistent, no duplicates")
        print(f"  {test_name}: ✅ PASSED")
        return True


# ════════════════════ 主测试 ════════════════════

if __name__ == "__main__":
    device = "npu"
    NUM_SMS = 8
    ITERS = 8

    print("=" * 60)
    print("CAS Claim Demo — Ascend NPU")
    print(f"NUM_SMS = {NUM_SMS}")
    print("=" * 60)

    # ── Test 1: CAS 同一地址 ──
    print(f"\n[Test 1] CAS same-address ({NUM_SMS} SM × {ITERS} tries)")
    c1 = torch.zeros(1, dtype=torch.int32, device=device)
    r1 = torch.full((NUM_SMS * ITERS,), -1, dtype=torch.int32, device=device)
    try:
        cas_same_addr_counter[(NUM_SMS, 1, 1)](
            c1, r1, NUM_SMS=NUM_SMS, ITERS=ITERS,
        )
        torch.npu.synchronize()
        vals1 = sorted([v for v in r1.cpu().tolist() if v >= 0])
        expected = NUM_SMS * ITERS
        dup = len(vals1) - len(set(vals1))
        print(f"  counter={c1.item()}, claimed={len(vals1)}, dup={dup}")
        print(f"  first 20: {vals1[:20]}")
        if dup == 0:
            print(f"  Test 1: ✅ CAS same-addr OK")
        else:
            print(f"  Test 1: ❌ dup={dup} — CAS same-addr NOT atomic "
                  f"(expected, same as ADD)")
    except Exception as e:
        msg = str(e)[:300]
        print(f"  Test 1: ❌ {msg}")

    # ── Test 2: CAS scatter 不同地址 ──
    TOTAL2 = NUM_SMS * 12
    MAX2 = TOTAL2 // NUM_SMS + 1
    print(f"\n[Test 2] CAS scatter ({NUM_SMS} SM, {TOTAL2} tasks)")
    ts2 = torch.zeros(TOTAL2, dtype=torch.int32, device=device)
    r2 = torch.full((NUM_SMS * MAX2,), -1, dtype=torch.int32, device=device)
    try:
        cas_scatter[(NUM_SMS, 1, 1)](
            ts2, r2, NUM_SMS=NUM_SMS, TOTAL=TOTAL2,
            MAX_PER_SM=MAX2,
        )
        torch.npu.synchronize()
        check_results(ts2, r2, MAX2, TOTAL2, NUM_SMS, "Test 2")
    except Exception as e:
        msg = str(e)[:300]
        print(f"  Test 2: ❌ {msg}")

    # ── Test 3: 探针 + CAS 不同地址 ──
    # TOTAL 必须 2 的幂 (用 & mask 代替 % modulo 避免编译卡死)
    TOTAL3 = 256
    PROBE = 2
    STAGGER = 3   # >= PROBE, 确保不同 SM 首轮区间不重叠
    ROUNDS = 20   # 每 SM 最多 claim 2*20=40 个, ×8 SM = 320 够覆盖 256
    MAX3 = ROUNDS * PROBE
    print(f"\n[Test 3] Probe + CAS claim "
          f"(TOTAL={TOTAL3}, PROBE={PROBE}, STAGGER={STAGGER}, ROUNDS={ROUNDS})")
    ts3 = torch.zeros(TOTAL3, dtype=torch.int32, device=device)
    c3 = torch.zeros(1, dtype=torch.int32, device=device)
    r3 = torch.full((NUM_SMS * MAX3,), -1, dtype=torch.int32, device=device)
    try:
        t0 = time.perf_counter()
        probe_cas_claim[(NUM_SMS, 1, 1)](
            ts3, c3, r3,
            TOTAL=TOTAL3, NUM_SMS=NUM_SMS,
            PROBE=PROBE, STAGGER=STAGGER,
            ROUNDS=ROUNDS, MAX_CLAIMED=MAX3,
        )
        torch.npu.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  elapsed: {elapsed*1000:.1f}ms  counter={c3.item()}")
        check_results(ts3, r3, MAX3, TOTAL3, NUM_SMS, "Test 3")
    except Exception as e:
        msg = str(e)[:300]
        print(f"  Test 3: ❌ {msg}")

    print("\n" + "=" * 60)
    print("Done.")
