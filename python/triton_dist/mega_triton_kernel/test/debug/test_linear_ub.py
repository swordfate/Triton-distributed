"""
独立测试: 只测线性 kernel, 用 NUM_SMS=1/2/4 对应的参数
目的: 隔离出是单个 kernel 的问题, 还是多 kernel 组合的问题
"""

import torch
import triton
import triton.language as tl
from torch_npu.contrib import transfer_to_npu

try:
    from triton.language.extra.cann import extension as al
except ImportError:
    al = None
    print("No al")


@triton.jit
def linear_kernel_test(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    SUB_BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, NUM_SMS: tl.constexpr,
):
    """从 megakernel 的 linear_task_compute 精简而来"""
    pid_m = tl.program_id(0)
    start_m = pid_m * BLOCK_SIZE_M
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)

    num_pid_n = N // BLOCK_SIZE_N
    for pid_n in range(num_pid_n):
        base_start_n = pid_n * BLOCK_SIZE_N
        k_tiles = K // BLOCK_SIZE_K

        for i in tl.range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N, num_stages=NUM_STAGES):
            start_n = base_start_n + i
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


# ─── 测试参数: 模拟不同 NUM_SMS 下的 QKVProj ───
# NUM_SMS=1: SUB=416, BLOCK_SIZE_N=6240  (15 sub-blocks, 1 tile)
# NUM_SMS=2: SUB=400, BLOCK_SIZE_N=3200  (8 sub-blocks, 2 tiles)
# NUM_SMS=4: SUB=400, BLOCK_SIZE_N=1600  (4 sub-blocks, 4 tiles)
# MLPFC1 NUM_SMS=4: SUB=416, BLOCK_SIZE_N=6240 (15 sub-blocks, 4 tiles) — 这能过!

TEST_CASES = [
    # (name, SUB, BLOCK_SIZE_N, NUM_STAGES)
    ("qkv_nsms1_sub416_bn6240", 416, 6240, 4),     # NUM_SMS=1 QKVProj — 崩
    ("qkv_nsms2_sub400_bn3200", 400, 3200, 4),     # NUM_SMS=2 QKVProj — 崩
    ("qkv_nsms4_sub400_bn1600", 400, 1600, 4),     # NUM_SMS=4 QKVProj — OK
    ("mlp_nsms4_sub416_bn6240", 416, 6240, 4),     # NUM_SMS=4 MLPFC1 — OK (同样的 sub=416!)
    ("sub416_bn1248",           416, 1248, 4),      # 小 BLOCK_SIZE_N 对照组
    ("sub320_bn1280",           320, 1280, 4),      # 小 SUB 对照组
]


def run_test(name, SUB, BLOCK_SIZE_N, NUM_STAGES, device='npu'):
    M, N, K = 1, 6144, 4096  # QKVProj 尺寸
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_K = 256
    NUM_SMS = 1
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    c = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

    try:
        linear_kernel_test[(num_pid_m, 1, 1)](
            a, b, c,
            M=M, N=N, K=K,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            SUB_BLOCK_SIZE_N=SUB, BLOCK_SIZE_K=BLOCK_SIZE_K,
            NUM_STAGES=NUM_STAGES, NUM_SMS=NUM_SMS,
        )
        torch.npu.synchronize()
        print(f"  {name}: OK (c_sum={c.sum().item():.2f})")
    except Exception as e:
        msg = str(e)
        if "SIGSEGV" in msg or "Segmentation" in msg:
            print(f"  {name}: SIGSEGV")
        elif "MLIR" in msg:
            print(f"  {name}: MLIR CRASH")
        elif "507015" in msg or "aicore" in msg:
            print(f"  {name}: AICORE RUNTIME ERROR")
        else:
            print(f"  {name}: {msg[:120]}")


if __name__ == "__main__":
    import sys
    device = 'npu' if len(sys.argv) < 2 else sys.argv[1]
    print(f"=== Linear kernel UB test on {device} ===")
    for name, sub, bn, ns in TEST_CASES:
        run_test(name, sub, bn, ns, device)
