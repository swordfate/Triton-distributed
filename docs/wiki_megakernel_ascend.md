# Megakernel 昇腾迁移与优化

> 将 ByteDance Seed Triton-distributed 的 megakernel 从 NVIDIA GPU 迁移至 Ascend 910B3 NPU，并进行性能调优。
> 目标模型：Qwen3-8B（decode 场景），覆盖单卡功能、单卡性能、多卡分布式、动态调度探索。

---

## 一、背景

### 1.1 项目来源

[Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) 是字节跳动 Seed 团队开源的分布式编译框架，基于 OpenAI Triton。其核心创新之一是 **MegaTritonKernel（megakernel）**：将完整的 Transformer 推理图（多层的 attention + MLP + 通信）编译为**一次 GPU kernel launch**，通过 persistent kernel + scoreboard 依赖管理消除传统逐 op 调度中的 kernel launch 开销和 HBM 读写冗余。

### 1.2 迁移目标

- **硬件平台**：Ascend 910B3（20 个 AICore，每个含 1 个 Cube + 2 个 Vector）
- **软件栈**：triton-ascend（华为开源 Triton 昇腾后端）、bishengir-compile + hivmc 编译链、CANN 8.5.0
- **目标模型**：Qwen3-8B（decode 场景，batch_size=1，input_len=8000，gen_len=192）
- **目标性能指标**：TPOT（Time Per Output Token）

### 1.3 GPU vs NPU 关键编译链差异

| | NVIDIA GPU | Ascend NPU |
|---|---|---|
| Triton 后端 | Triton-CUDA (PTX) | triton-ascend (ttadapter IR) |
| IR 转换 | TTIR → TritonGPU IR → PTX | TTIR → ttadapter IR (Linalg+HIVM+HFusion) |
| 后端编译器 | ptxas / nvcc | bishengir-compile → hivmc |
| 内存模型 | `ld.global.require.gpu` 单指令保序 | 无等效指令，需组合方案 |
| 原子指令 | 硬件 RMW（L2 cache 层串行化） | 软件模拟（load + add + store） |
| triton-ascend 版本 | - | 3.2.0（基于 Triton 3.2，语法覆盖 ~85%） |

---

## 二、单卡迁移

### 2.1 功能方面 —— 昇腾解决方案

#### 2.1.1 语法与编译兼容

**不支持或受限的 Triton 语法**：

| 语法 | 状态 | 替代方案 |
|------|------|----------|
| `while` / `break` / `continue` | 不支持 | `tl.static_range` 编译期展开，或 `for-range`（`scf.for`）|
| `tl.static_range` 的 start/step 参数 | 必须为 constexpr | 运行时起始值用 `for-range` + 条件分支 |
| 模运算 `%` | bishengir-compile 编译卡死（展开为除法微码） | 改用 `& (N-1)` 掩码，要求 `N` 为 2 的幂 |

**libdevice 函数不兼容**：

CUDA 版本使用了 `__nv_fast_expf` 等 GPU libdevice 内置函数，这些函数在昇腾上不存在。解决方案：替换为纯 Triton 实现或使用 CANN 提供的 bitcode 库链接。

#### 2.1.2 计分板跨核同步

GPU 上使用 `ld.global.require.gpu` 单条 PTX 指令同时实现**缓存一致性**和**访存保序**。昇腾 AICore 硬件不支持等效单指令，需用组合方案：

| 需求 | GPU 方案 | Ascend 替代方案 |
|------|----------|-----------------|
| 缓存一致性 | `ld.global.require.gpu` | `tl.load(…, cache_modifier=".cv", volatile=True)` |
| AIV ↔ AIC 内存屏障 | 同上 | `tl.sync_block_set('vector', 'cube', id)` + `tl.sync_block_wait('cube', 'vector', id)` |
| AIV ↔ AIV 内存屏障 | 同上 | `BAR.ALL` 内联汇编 |

**计分板等待实现**（`scoreboard_wait_deps_flat`）：

```python
for i_base in range(0, num_signals, 128):
    offsets = i_base + tl.arange(0, 128)
    mask = offsets < num_signals
    all_ready_val = 0
    while all_ready_val == 0:
        vals = tl.load(sb_wait_base_ptr + offsets, mask=mask,
                       other=TILE_READY_SIGNAL,
                       cache_modifier=".cv", volatile=True)
        min_val = tl.min(vals, axis=0)
        if min_val == TILE_READY_SIGNAL:
            all_ready_val = 1
```

**计分板释放实现**（`scoreboard_release_tile_flat`）：

```python
dummy = tl.arange(0, 1)
tl.inline_asm_elementwise(asm="BAR.ALL", ...)  # 释放前 barrier
tl.store(sb_ptr, TILE_READY_SIGNAL)            # 写信号
tl.inline_asm_elementwise(asm="BAR.ALL", ...)  # 释放后 barrier
```

#### 2.1.3 Profiling 适配

GPU 上使用 CUDA profiler 或 `tl.inline_asm`（PTX）。昇腾上通过**手写内联汇编**实现 intra-kernel 打点：

```python
tl.inline_asm_elementwise(
    asm="...",                    # 昇腾 AI Core 汇编
    constraints="=l,0",
    args=[dummy],
    dtype=tl.int32,
    is_pure=False,
    pack=1,
)
```

打点数据写入全局 profiler buffer，kernel 执行后导出为 Perfetto trace 进行可视化分析。

### 2.2 性能方面 —— 昇腾性能调优

#### 2.2.1 计算效率优化

**Grid 划分优化**：

每个 op 的 tile 分配到 20 个 AICore 上执行。通过调整 tile 大小（`BLOCK_SIZE_M`、`BLOCK_SIZE_N`、`SUB_BLOCK_SIZE_N`）减少单 op 的总 tile 数，降低 work queue 解码开销和 scoreboard 同步开销。例如 QKVProj 的 tile 配置：

```
NUM_SMS=1: SUB=416, BLOCK_SIZE_N=6240  (15 sub-blocks, 1 tile)
NUM_SMS=4: SUB=400, BLOCK_SIZE_N=1600  (4 sub-blocks, 4 tiles)
MLPFC1:    SUB=208, BLOCK_SIZE_N=1248  (NUMS_SMS=20, tile≤BLOCK_N)
```

**访存合并**：

将 `tl.reshape` → `tl.permute` → `tl.split` 多步串行读写合并为 `tl.extract_slice` 一步完成：

```python
# 优化前（三步，每步读写一次 HBM）
reshaped = tl.reshape(qkv, (bs, seq, num_heads, head_dim))
permuted = tl.permute(reshaped, (0, 2, 1, 3))
q, k, v = tl.split(permuted, [q_heads, kv_heads, kv_heads], dim=1)

# 优化后（一步 extract_slice，一次 HBM 读写）
q = tl.extract_slice(qkv, (0, 0, 0, 0), (bs, seq, q_heads, head_dim), (1, 1, 1, 1))
```

#### 2.2.2 访存优化

**KV Cache Page 级大粒度加载**：

原始方案逐 token 加载 KV cache（MTE2 对小粒度访问延迟高）。改为按 page（64 tokens）大粒度批量加载：一次 `tl.load` 加载整个 page，提高 MTE2（DMA Engine 2）带宽利用率。

**Weight 预取到 L1 Cache**：

利用 `prefetch_task_compute` 将权重数据预取到 AICore 的 L1 cache，减少 compute 阶段从 HBM 读取权重的延迟。

#### 2.2.3 计分板同步延迟优化

**问题发现**：

通过 bench 测试发现：AICore 数量从 4 增加到 20 时，计分板等待延迟从 ~2μs 上升到 ~30μs。对比 H800 在任意 SM 数量下延迟始终 <1μs。

**根因分析**：

每个 AICore 的计分板等待是**单线程串行 polling**——逐个检查 scoreboard 槽位的信号值。当 AICore 数量增加、每个槽的写者增多时，需要等待的信号总量增大，单线程 polling 无法并行加速。

**解决方案**：

1. **向量并行 polling**（128 路）：将等待逻辑展开为 128 个元素并行 `tl.load` + `tl.min`，一次性检查 128 个信号

```python
for i_base in range(0, num_signals, 128):
    offsets = i_base + tl.arange(0, 128)
    vals = tl.load(sb_ptr + offsets, mask=mask, cache_modifier=".cv", volatile=True)
    min_val = tl.min(vals, axis=0)  # 128 路并行比较
```

2. **计分板 cacheline 对齐**：`MAX_NUM_TILES_PER_OP = (MAX_NUM_TILES_PER_OP + 127) // 128 * 128`，确保每行的信号槽在 128B cacheline 边界对齐，减少跨 cacheline 访问延迟

**效果**：等待延迟从 ~30μs 降至 ~5μs 级别。

---

## 三、多卡分布式

### 3.1 功能方面 —— 昇腾解决方案

#### 3.1.1 依赖基础

分布式能力依赖以下组件：

- **triton-distributed-ascend**（字节 2012 实验室提供）：shmem 分布式原语（`dl.notify`、`dl.wait`、`dl.symm_at`、`dl.consume_token`）
- **NPU-IR**：Ascend NPU 编译 IR 能力支持
- **ash（Ascend SHMEM）**：基于 CANN 的对称内存分配接口（`ash.aclshmem_create_tensor`）

#### 3.1.2 Allreduce 算子实现

**算法设计**：基于 Reduce-Scatter + All-Gather 两阶段协议：

```
Phase 1 (Reduce-Scatter):
  1. Notify peers: 通知所有 rank 当前 rank 数据就绪
  2. Wait: 等待所有 rank 数据就绪 (dl.wait)
  3. Reduce: 用 dl.symm_at 远程读取所有 rank 的局部结果 → 累加 → 写入本地 RS buffer
  4. Intra-node sync: AICore 间同步

Phase 2 (All-Gather):
  5. Notify: 通知所有 rank RS buffer 就绪
  6. Wait: 等待所有 rank RS buffer 就绪
  7. Gather: 从远程 RS buffer 拉取数据写入 C 矩阵
```

**通信原语说明**：

| 原语 | 功能 | 语义 |
|------|------|------|
| `dl.notify(ptr, rank, signal=1)` | 点对点通知 PE `rank` 的地址 `ptr` 信号置 1 | 非集体操作，支持 per-tile 使用 |
| `dl.wait(ptr, n, waitValue=1)` | 等待本地地址 `ptr` 开始的 `n` 个连续槽位达到 `waitValue` | 可等待任意数量信号 |
| `dl.symm_at(ptr, rank)` | 获取 PE `rank` 上 `ptr` 的远端地址 | 返回可加载的远端指针 |
| `dl.consume_token(ptr, token)` | 用 token 授权远端读 | token 来自 `dl.wait` 返回值 |

**性能测试**：独立 allreduce kernel 的带宽和延迟数据（待填入实际测试数据）。

#### 3.1.3 Allreduce 集成到 Megakernel

**Barrier 分配**：

Barrier tensor 通过 `ash.aclshmem_create_tensor` 分配在对称共享内存中，确保所有 rank 可互相访问。per-tile 布局为 `[num_tiles × rank_size]` 个 int64 槽位。

**易错点 1：Barrier 清零**：

每次 megakernel 执行后必须将 barrier tensor 清零，否则残留信号值会影响下一轮 token 的通信同步：

```python
# model_builder.py run()
if self.world_size > 1:
    for idx in range(self.model.num_layers):
        self.model.layers[idx].attn.barrier_tensor.zero_()
        self.model.layers[idx].mlp.barrier_tensor.zero_()
    self.model.barrier_tensor.zero_()
```

**易错点 2：依赖链建立**：

Megakernel 框架通过 `Graph.to_tasks()` 按 tensor shape 和 data_ptr 计算生产者-消费者依赖。通信算子（allreduce）的输入输出需要正确设置 `InputDependencyDesc` 和 `OutputTilingDesc`，否则框架无法建立正确的 tile 级依赖。

### 3.2 性能方面

#### 3.2.1 优势

**Linear 算子分核后性能良好**：

Linear/FC/QKV/O-proj 等矩阵乘算子的 tile 划分与 AICore 数量（20）对齐，每个 AICore 处理的 tile 数量和计算量均匀，AICore 利用率高。

**Per-tile Allreduce 支持 tile 级计算-通信重叠**：

将 AR 从 bulk 两阶段协议改为 per-tile 独立协议后，每个 AR tile 只依赖对应的 compute tile（而非所有 compute tile）。通过 scoreboard 依赖管理，AICore 可在完成一个 compute tile 后立即启动对应的 AR tile，实现计算与通信的流水化。

#### 3.2.2 待解决问题

**Attention 算子 AICore 浪费**：

Flash decode 的 split/combine 模式在分核时无法充分利用 20 个 AICore——head 数量（8 个 KV head × 1 个 q head per KV head = 8 个 group）有限，总 tile 数不足以覆盖所有 core，部分 AICore 空闲。

**`sub_vec_id` 导致 AIV 浪费**：

Allreduce 和 profiling 中使用了 `sub_vec_id()` 来区分 Vector core（AIV）。triton-ascend 编译器检测到 `sub_vec_id` 后会关闭自动 AIV 分配优化。除非手动使用 `tl.parallel` 等方式显式指定 AIV 并行，否则存在 AIV 计算资源浪费。

---

## 四、昇腾 Triton-Ascend Megakernel 问题汇总

### 4.1 静态调度问题

| # | 类别 | 问题描述 | 根因 | 解决方案 |
|---|------|----------|------|----------|
| 1 | 语法 | `while` / `break` / `continue` 不支持 | triton-ascend 3.2.0 基于 Triton 3.2，语法覆盖 ~85% | 用 `tl.static_range`（编译期展开）或 `for-range`（`scf.for`）替代控制流 |
| 2 | 语法 | `tl.static_range(start, end, step)` 的 start/step 必须为 constexpr | Triton 编译期展开要求所有边界为常量 | 运行时起始值改用 `for-range` + 条件分支 |
| 3 | 编译 | `__nv_fast_expf` 等 libdevice 内置函数无法编译 | GPU 专属，昇腾无等效内置函数 | 替换为纯 Triton 逐元素实现，或通过 `--link-aicore-bitcode` 链接 CANN 提供的 bitcode 库 |
| 4 | 编译 | `%` 模运算导致 bishengir-compile 编译阶段卡死 | bishengir-compile 将模运算展开为除法微码序列，导致编译超时或死循环 | 改用 `& (N-1)` 掩码替代 `% N`，要求除数 N 为 2 的幂 |
| 5 | 编译 | `scf.while` + `hivm.hir.store atomic=<add>` 组合 → bishengir SIGSEGV 崩溃 | bishengir-compile 内部对 while 循环中的原子 store 处理存在 bug | 避免 while 内使用 atomic 操作；改用 for-range 或 chunk 方案 |
| 6 | 编译 | `scf.for` + 原子操作导致多个 block 被串行调度 | bishengir-compile/hivmc 对 `scf.for` 内的原子 store 做了保守的 block 调度优化 | 同 #5——避免 for 循环内使用同一地址的原子操作 |
| 7 | 内存模型 | GPU 的 `ld.global.require.gpu` 在昇腾无等效单指令 | AICore 指令集不同，无 GPU 的全局内存获取-释放语义指令 | 组合方案：`tl.load(cache_modifier=".cv", volatile=True)` + `sync_block_set/wait` + `BAR.ALL` |
| 8 | 内存模型 | Cube 写 → Vector 读的数据可见性无硬件保证 | Cube 和 Vector 使用独立的 M-Queue 和 V-Queue，无自动保序 | `tl.sync_block_set('vector', 'cube', event_id)` / `tl.sync_block_wait('cube', 'vector', event_id)` |
| 9 | 内存模型 | 多个 Vector Core 之间的 store/load 顺序无硬件保序 | Vector Core 间无自动缓存一致性 | `BAR.ALL` 内联汇编 barrier |
| 10 | Profiling | CUDA profiler / PTX profiler 在昇腾不可用 | 昇腾无等效 profiler 工具 | 手写 `tl.inline_asm_elementwise` 实现 kernel 内时间打点，导出 Perfetto trace |
| 11 | 性能 | 计分板等待延迟随 AICore 数量非线性增长（20 SM 时 ~30μs，H800 始终 <1μs） | 单线程串行 polling 信号槽，等待时间与信号数量成正比 | ① 128 路向量并行 polling（`tl.arange(0, 128)` + `tl.min`）；② 计分板每行 cacheline 对齐（128B 边界） |
| 12 | 性能 | KV cache 逐 token 访存 MTE2 效率低 | MTE2 对小粒度（<64B）访问延迟高 | 按 page（64 tokens × head_dim）大粒度加载 |
| 13 | 性能 | 多步读写产生冗余指令 | triton-ascend 对 `reshape + permute + split` 生成多条 load/store 指令 | 使用 `tl.extract_slice` 一步替代 |
| 14 | 分核 | `sub_vec_id()` 导致 triton-ascend 关闭自动 AIV 分配 | 编译器检测到 `sub_vec_id` 后不做自动 AIV 2 路并行优化 | 手动使用 `tl.parallel(0, 2, bind_sub_block=True)` 显式指定 AIV 并行 |
| 15 | 分核 | Attention 算子（split/combine）AICore 利用率低 | Head 数量有限，总 tile 数不足覆盖全部 20 个 AICore | 待优化（KV_SPLITS 自适应增大、tile 大小调整等） |

### 4.2 动态调度问题

#### 4.2.1 GPU 方案回顾

GPU 上 megakernel 使用 `tl.atomic_add` 实现 per-task 动态竞争分配：每个 SM 在循环中调用 `atomic_add(counter, 1)` 获取下一个 task ID，处理完后再抢下一个。GPU 的 `atomic_add` 是硬件 RMW 指令——L2 cache 层锁定、读值、修改、写回、解锁、返回旧值，整个过程对所有 SM 串行化，保证互斥。

#### 4.2.2 NPU 上 atomic_add 的失效分析

在 Ascend NPU 上对 `atomic_add` 进行了系统性验证（demo 1-18，`test_atomic_demo*.py`）：

| Demo | 模式 | 结果 | 分析 |
|------|------|------|------|
| 1 | 单次 `atomic_add`（无循环，4 SM × 1 call） | ✅ 正确 | SM 启动时序分散，load-store 窗口不重叠 |
| 2 | `while` + `atomic_add`（4 SM，while 条件判断） | ❌ bishengir SIGSEGV | 编译器无法处理 while 内的原子操作 |
| 5 | `static_range` 展开 + `atomic_add`（4 SM × 8 背靠背） | ❌ 大量重复值（unique: 18/32） | 展开无循环，4 SM 真正并发但锁步执行：所有 SM 同时 load 到相同 counter 值，RMW 不互斥 |
| 5.5 | `for-range` + `atomic_add`（20 SM × 10 iter） | counter=200, vals 完全顺序 `[0..199]` | `scf.for` 触发 block 串行调度，20 SM 顺序执行而非并发 |
| 9 | `for-range` + `atomic_add`（4 SM × 27 iter） | counter=100, vals 完全顺序 `[0..99]` | 同 #5.5，block 串行 |
| CAS | `atomic_cas` 同一地址（`static_range` 展开） | ❌ 大量重复值 | CAS 和 ADD 走同一条 `hivm.hir.store atomic=<op>` 路径 |
| CAS | `atomic_cas` 不同地址（scatter 模式） | ✅ 正确认领 | 无竞争，不同地址独立 |

**根因**：Ascend AICore 指令集中**没有硬件 RMW 原子指令**。Triton 的 `tl.atomic_add` 在 IR 层面降级为：

```mlir
memref.copy %counter, %alloc          // ① 普通 load（读旧值）
hivm.hir.store ... atomic = <add>     // ② 原子 store（累加）
%result = tensor.extract %alloc       // ③ 返回旧值（来自步骤①）
```

步骤①和②之间有 gap。在锁步竞争下（`static_range` 展开），所有 SM 同时执行步骤①读到相同值 → 同时执行步骤②写入相同的 counter+1 → 返回相同的"旧值"。

#### 4.2.3 当前方案：Chunk 调度

受限于上述约束，当前采用 Chunk 调度：

- 每个 SM 在循环**外**调用一次 `atomic_add(counter, CHUNK_SIZE)`，获取一段连续 task 区间
- 循环**内**执行 chunk 中的 task，无需原子操作
- task 执行时间差异打破 SM 间的锁步，单次 `atomic_add` 的返回值在非锁步下可靠

**局限性**：Chunk 大小固定（受限于可展开层数），不能实现完全动态的 per-task load balancing。

---

## 五、总结与展望

### 5.1 当前状态

| 维度 | 状态 |
|------|------|
| 单卡功能 | ✅ Qwen3-8B decode 跑通 |
| 单卡性能 | ✅ TPOT 达到可用水平（待填具体数据） |
| 多卡分布式 (TP=2) | ✅ Allreduce 集成完成 |
| 动态调度 | ⚠️ Chunk 方案可用，真动态调度受限于硬件 |

### 5.2 与 GPU 方案的关键差距

| | GPU (H800) | Ascend (910B3) |
|---|---|---|
| 同步延迟 | <1μs（任意 SM 数） | ~5μs（优化后，仍高于 GPU） |
| 原子操作 | 硬件 RMW 互斥 | 软件模拟，无互斥 |
| 动态调度 | per-task 竞争 | Chunk 调度 |
| Profiling | 原生 CUDA profiler | 手写 asm 打点 |

### 5.3 后续优化方向

1. **Attention 分核**：改进 flash decode split/combine 的 tile 策略，提高 AICore 利用率
2. **AIV 利用率**：在 allreduce/profiling 中使用 `tl.parallel` 显式指定 AIV 并行，替代 `sub_vec_id`
3. **动态调度**：探索 per-tile scoreboard + 去中心化 CAS 认领的替代方案
4. **计算-通信重叠**：per-tile allreduce + 细粒度 scoreboard 依赖实现 tile 级流水化
