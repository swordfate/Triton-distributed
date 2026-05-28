"""
Demo 15-18: 逐步叠加 megakernel 中的真实操作, 定位 Bisheng 崩溃条件

已验证: atomic_add 单独 OK, scope+scoreboard 单独 OK (static megakernel)
嫌疑: 特定组合触发崩溃

15. atomic_add 单独 (无 scope, 无 loop, 无 BAR/SYNC)
16. atomic_add + al.scope(core_mode="vector") 包裹
17. atomic_add + al.scope + for-range(chunk) + SCOREBOARD_WAIT
18. atomic_add + al.scope + for-range(chunk) + SCOREBOARD_WAIT + BAR.ALL + sync_block_set/wait
     ↑ 和 megakernel 最接近的简化版
"""

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cann import extension as al
    tl.sync_block_set = al.sync_block_set
    tl.sync_block_wait = al.sync_block_wait
    tl.extract_slice = al.extract_slice
    _HAS_AL = True
except ImportError:
    _HAS_AL = False
    print("WARNING: al module not available, using simplified scope emulation")


# ═══ Demo 15: atomic_add 单独 ═══
@triton.jit
def demo15(counter, result, NUM_SMS: tl.constexpr):
    pid = tl.program_id(0)
    cur = tl.atomic_add(counter, 1)
    tl.store(result + pid, cur)


# ═══ Demo 16: atomic_add + al.scope ═══
@triton.jit
def demo16(counter, result, NUM_SMS: tl.constexpr):
    pid = tl.program_id(0)
    if _HAS_AL:
        with al.scope(core_mode="vector"):
            cur = tl.atomic_add(counter, 1)
    else:
        cur = tl.atomic_add(counter, 1)
    tl.store(result + pid, cur)


# ═══ Demo 17: atomic_add + al.scope + for-range(chunk) + scoreboard spin ═══
@triton.jit
def demo17(
    counter, result, scoreboard,
    NUM_SMS: tl.constexpr, TOTAL: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    if _HAS_AL:
        with al.scope(core_mode="vector"):
            my_start = tl.atomic_add(counter, CHUNK)
    else:
        my_start = tl.atomic_add(counter, CHUNK)

    if my_start < TOTAL:
        for i in range(CHUNK):
            cur = my_start + i
            if cur < TOTAL:
                # 模拟 scoreboard wait (和 megakernel 完全一致的写法)
                TILE_READY = 2
                sb_ptr = scoreboard + cur
                ready = 0
                while ready == 0:
                    val = tl.load(sb_ptr, cache_modifier=".cv", volatile=True).to(tl.int32)
                    if val == TILE_READY:
                        ready = 1

                tl.store(result + cur, pid)

                # 模拟 scoreboard release
                dummy = tl.arange(0, 1)
                tl.inline_asm_elementwise(
                    asm="BAR.ALL",
                    constraints="=l,0", args=[dummy],
                    dtype=tl.int32, is_pure=False, pack=1,
                )
                tl.store(sb_ptr, TILE_READY)


# ═══ Demo 18: 和 megakernel 最接近的简化版 ═══
# atomic_add + al.scope + for-range(chunk) + scoreboard_wait_flat +
# BAR.ALL + sync_block_set/wait (模拟 AIC task 的访存保序)
@triton.jit
def demo18(
    counter, result, scoreboard,
    NUM_SMS: tl.constexpr, TOTAL: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    TILE_READY: tl.constexpr = 2

    if _HAS_AL:
        with al.scope(core_mode="vector"):
            my_start = tl.atomic_add(counter, CHUNK)
    else:
        my_start = tl.atomic_add(counter, CHUNK)

    if my_start < TOTAL:
        for i in range(CHUNK):
            cur = my_start + i
            if cur < TOTAL:
                # wait_deps (同 megakernel)
                sb_ptr = scoreboard + cur
                ready = 0
                while ready == 0:
                    val = tl.load(sb_ptr, cache_modifier=".cv", volatile=True).to(tl.int32)
                    if val == TILE_READY:
                        ready = 1

                tl.store(result + cur, pid)

                # 访存保序: 模拟 AIC task 的 sync 路径 (同 megakernel)
                task_type = 6  # 假 task type, 触发 AIC sync 路径
                if task_type in (1, 3, 4, 7, 9, 11):
                    if _HAS_AL:
                        with al.scope(core_mode="vector"):
                            tl.sync_block_set('vector', 'cube', 5)
                        with al.scope(core_mode="cube"):
                            tl.sync_block_wait('vector', 'cube', 5)
                else:
                    dummy = tl.arange(0, 1)
                    tl.inline_asm_elementwise(
                        asm="BAR.ALL",
                        constraints="=l,0", args=[dummy],
                        dtype=tl.int32, is_pure=False, pack=1,
                    )

                # release (同 megakernel)
                if _HAS_AL:
                    with al.scope(core_mode="vector"):
                        dummy2 = tl.arange(0, 1)
                        tl.inline_asm_elementwise(
                            asm="BAR.ALL",
                            constraints="=l,0", args=[dummy2],
                            dtype=tl.int32, is_pure=False, pack=1,
                        )
                        tl.store(sb_ptr, TILE_READY)
                        dummy3 = tl.arange(0, 1)
                        tl.inline_asm_elementwise(
                            asm="BAR.ALL",
                            constraints="=l,0", args=[dummy3],
                            dtype=tl.int32, is_pure=False, pack=1,
                        )


def run(name, fn, **kw):
    try:
        fn[(4, 1, 1)](**kw)
        torch.npu.synchronize()
        print(f"  {name}: OK")
    except Exception as e:
        msg = str(e)
        if "SIGSEGV" in msg or "Segmentation" in msg:
            print(f"  {name}: SIGSEGV")
        elif "MLIR" in msg:
            print(f"  {name}: MLIR CRASH")
        else:
            print(f"  {name}: {msg[:120]}")


if __name__ == "__main__":
    device = 'npu'
    N = 4
    T = 8
    C = T // N

    print("=== Bisheng crash trigger bisection ===")
    print(f"al.scope available: {_HAS_AL}")

    c15 = torch.zeros(1, dtype=torch.int32, device=device)
    r15 = torch.full((N,), -1, dtype=torch.int32, device=device)
    run("15: atomic_add alone           ", demo15, counter=c15, result=r15, NUM_SMS=N)

    c16 = torch.zeros(1, dtype=torch.int32, device=device)
    r16 = torch.full((N,), -1, dtype=torch.int32, device=device)
    run("16: + al.scope                 ", demo16, counter=c16, result=r16, NUM_SMS=N)

    c17 = torch.zeros(1, dtype=torch.int32, device=device)
    r17 = torch.full((T,), -1, dtype=torch.int32, device=device)
    s17 = torch.zeros(T, dtype=torch.int32, device=device)
    run("17: + for-range + scoreboard   ", demo17,
        counter=c17, result=r17, scoreboard=s17, NUM_SMS=N, TOTAL=T, CHUNK=C)

    c18 = torch.zeros(1, dtype=torch.int32, device=device)
    r18 = torch.full((T,), -1, dtype=torch.int32, device=device)
    s18 = torch.zeros(T, dtype=torch.int32, device=device)
    run("18: + sync_block + BAR.ALL     ", demo18,
        counter=c18, result=r18, scoreboard=s18, NUM_SMS=N, TOTAL=T, CHUNK=C)
