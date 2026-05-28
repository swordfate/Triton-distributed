from typing import List, Dict, Tuple
import textwrap
from dataclasses import dataclass
from .registry import registry
from .task_base import TaskBase, CodeGenKey


@dataclass
class CodeGenOptions:
    """代码生成选项, 对齐 main 分支"""
    enable_profiling: bool = False
    enable_runtime_scheduler: bool = False
    enable_task_prefetch: bool = False
    target_hw: str = 'gpu'


def make_mega_kernel_src(tasks_dispatch_code: str, codegen_options: CodeGenOptions,
                         task_types_and_str: Dict[int, str]) -> str:
    """
        max_task_type: profiling use only
    """
    enalbe_profiling = codegen_options.enable_profiling
    target_hw = codegen_options.target_hw
    enable_runtime_scheduler = codegen_options.enable_runtime_scheduler
    enable_task_prefetch = codegen_options.enable_task_prefetch

    max_task_type = 0
    for k, v in task_types_and_str.items():
        max_task_type = max(max_task_type, k)
    scoreboard_wait_deps_task_type = max_task_type + 1
    task_decoding_task_type = scoreboard_wait_deps_task_type + 1
    load_before_wait_type = task_decoding_task_type + 1
    task_types_and_str[scoreboard_wait_deps_task_type] = "scoreboard_wait_deps"
    task_types_and_str[task_decoding_task_type] = "task_decoding"
    task_types_and_str[load_before_wait_type] = "load_before_wait"

    if target_hw == 'gpu':
        src = f"""
import triton
import triton.language as tl
from mega_triton_kernel_ascend.kernels_for_gpu import *

from mega_triton_kernel_ascend.kernels_for_gpu.task_context import Scoreboard
from mega_triton_kernel_ascend.tools.profiler import Profiler
# from triton.language.extra.cuda.language_extra import tid
@triton.jit
def MEGA_TRITON_KERNEL(
    {"profiler_buf, # ensor<uint64>" if enalbe_profiling else ""}
    work_queues, # [MAX_INS, NUM_SMS, INS], int32
    num_tasks_per_wq, #[num_sms,]
    scoreboard_ptr,
    task_deps_ptr,  # [num_deps_entry_of_all_tasks, INT_PER_DEPS]

    INT_PER_DEPS: tl.constexpr,
    INT_PER_TASK: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    num_warps: tl.constexpr
):
    {f"profiler = Profiler.create(profiler_buf, 0, is_leader=(tid(0) == 0), ENABLE_PROFILING={enalbe_profiling})" if enalbe_profiling else ""}

    WARP_SIZE: tl.constexpr = 32
    NUM_THREADS: tl.constexpr = num_warps * WARP_SIZE
    scoreboard = Scoreboard(task_deps_ptr, INT_PER_DEPS, scoreboard_ptr, MAX_TASK_ID, MAX_NUM_TILES_PER_OP, tl.constexpr(1), NUM_THREADS)
    sm_id = tl.program_id(axis=0)
    num_tasks = tl.load(num_tasks_per_wq + sm_id)
    offset = INT_PER_TASK * NUM_SMS

    TASK_TYPE_OFFSET = 0
    LAYER_ID_OFFSET = 1
    TASK_ID_OFFSET = 2
    TILE_ID_OR_START_OFFSET = 3
    DEPEND_ENTRY_START_OFFSET = 4
    DEPEND_ENTRY_END_OFFSET = 5
    IO_TENSORS_OFFSET = 6

    for i in range(num_tasks):
        task_type = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TASK_TYPE_OFFSET).to(tl.int32)
        layer_id = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + LAYER_ID_OFFSET).to(tl.int32)
        task_id = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TASK_ID_OFFSET).to(tl.int32)
        tile_id_or_start = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TILE_ID_OR_START_OFFSET).to(tl.int32)
        depend_entry_start = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + DEPEND_ENTRY_START_OFFSET).to(tl.int32)
        depend_entry_end = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + DEPEND_ENTRY_END_OFFSET).to(tl.int32)

        io_tensors_ptr = work_queues + i * offset + sm_id * INT_PER_TASK + IO_TENSORS_OFFSET
        task_base_info = TaskBaseInfo(io_tensors_ptr, layer_id, task_id, tile_id_or_start, depend_entry_start, depend_entry_end, MAX_NUM_TENSOR_DIMS)

        # task kernel need to set signal for each tile
        {f"profiler = profiler.record(is_start=True, task_type={scoreboard_wait_deps_task_type})" if enalbe_profiling else ""}
        scoreboard.wait_deps(task_base_info)
        {f"profiler = profiler.record(is_start=False, task_type={scoreboard_wait_deps_task_type})" if enalbe_profiling else ""}

        #### run task ####
        {"profiler = profiler.record(is_start=True, task_type=task_type)" if enalbe_profiling else ""}
{textwrap.indent(tasks_dispatch_code.strip(), '        ')}
        {"profiler = profiler.record(is_start=False, task_type=task_type)" if enalbe_profiling else ""}
"""
    else:
        aic = []
        for k,v in task_types_and_str.items():
            if v in ['QKVProjTask', 'PagedGQAFwdTask', 'OProjTask', 'MLPFC1Task', 'MLPFC2Task', 'LinearTask', 'AttnSplitTask']:
                aic.append(k)

        # ---- 按调度模式分叉: 静态调度 vs 动态调度 ----
        if enable_runtime_scheduler:
            # ===================================================================
            # 动态调度模式: 所有 task 放入一个全局平面队列 (num_sms=1),
            # 每个 SM 通过 atomic_add 竞争获取下一个 task, 空闲 SM 自动承接更多工作.
            # Ascend 适配: while+atomic_add 导致 Bisheng SIGSEGV, 改用 for-range.
            # ===================================================================
            work_queue_start_param = "work_queue_start, # [1,] int32 全局原子计数器, chunk 分配"
            task_fetch_body = _make_npu_dynamic_scheduler_body(
                aic=aic,
                enalbe_profiling=enalbe_profiling,
                tasks_dispatch_code=tasks_dispatch_code,
                scoreboard_wait_deps_task_type=scoreboard_wait_deps_task_type,
                load_before_wait_type=load_before_wait_type,
                enable_task_prefetch=enable_task_prefetch,
            )
        else:
            # ===================================================================
            # 静态调度模式: 每个 SM 有固定的 per-SM 工作队列 (round-robin 预分配)
            # ===================================================================
            work_queue_start_param = ""
            task_fetch_body = _make_npu_static_scheduler_body(
                aic=aic,
                enalbe_profiling=enalbe_profiling,
                tasks_dispatch_code=tasks_dispatch_code,
                scoreboard_wait_deps_task_type=scoreboard_wait_deps_task_type,
                load_before_wait_type=load_before_wait_type,
            )

        src = f"""
import triton
import triton.language as tl
from mega_triton_kernel_ascend.kernels_for_npu import *

from mega_triton_kernel_ascend.kernels_for_npu.task_context_utils import *
from mega_triton_kernel_ascend.tools.profiler import init_profiler, record_event
# from triton.language.extra.cuda.language_extra import tid

# for 新版分布式TA
from triton.language.extra.cann import extension as al
tl.sync_block_set = al.sync_block_set
tl.sync_block_wait = al.sync_block_wait
tl.extract_slice = al.extract_slice

@triton.jit
def MEGA_TRITON_KERNEL(
    {"profiler_buf, # ensor<uint64>" if enalbe_profiling else ""}
    {work_queue_start_param}
    work_queues, # [MAX_INS, NUM_SMS, INS], int32
    num_tasks_per_wq, #[num_sms,]
    scoreboard_ptr,
    task_deps_ptr,  # [num_deps_entry_of_all_tasks, INT_PER_DEPS]

    INT_PER_DEPS: tl.constexpr,
    INT_PER_TASK: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    num_warps: tl.constexpr,
    debug_counts,
):
    pid = tl.program_id(0)
    {f"prof_base_ptr, prof_offset, prof_stride = init_profiler(pid, profiler_buf, 0, 1, num_blocks=NUM_SMS, ENABLE_PROFILING={enalbe_profiling})" if enalbe_profiling else ""}

    # 1. 移除 Scoreboard 和 TaskBaseInfo 对象的实例化
    TILE_READY_SIGNAL: tl.constexpr = 2 # 定义信号值

    WARP_SIZE: tl.constexpr = 32
    NUM_THREADS: tl.constexpr = num_warps * WARP_SIZE
    sm_id = tl.program_id(axis=0)

    TASK_TYPE_OFFSET = 0
    LAYER_ID_OFFSET = 1
    TASK_ID_OFFSET = 2
    TILE_ID_OR_START_OFFSET = 3
    DEPEND_ENTRY_START_OFFSET = 4
    DEPEND_ENTRY_END_OFFSET = 5
    IO_TENSORS_OFFSET = 6

{textwrap.indent(textwrap.dedent(task_fetch_body).strip(), '    ')}
"""
    return src, task_types_and_str


def _make_npu_static_scheduler_body(aic, enalbe_profiling, tasks_dispatch_code,
                                     scoreboard_wait_deps_task_type, load_before_wait_type):
    """生成 NPU 静态调度 (per-SM queue) 的 task 循环体. 所有行以 0 空格为基准缩进."""
    # prof 字符串: 基准缩进 4 空格 (for 循环体层级), dedent+indent 后变为 8 空格
    P0 = '' if enalbe_profiling else None  # 无 profiling 时 prof 变量不起作用
    prof_load_begin = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=1, task_type={load_before_wait_type}, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_load_end   = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=0, task_type={load_before_wait_type}, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_wait_begin = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=1, task_type={scoreboard_wait_deps_task_type}, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_wait_end   = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=0, task_type={scoreboard_wait_deps_task_type}, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_task_begin = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=1, task_type=task_type, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_task_end   = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=0, task_type=task_type, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''

    # Triton 不支持 flat chained or, 必须嵌套括号: (A or (B or (C or D)))
    _parts = [f'(task_type=={x})' for x in aic]
    aic_cond = _parts[0]
    for p in _parts[1:]:
        aic_cond = f'({aic_cond} or {p})'

    body = f"""\
num_tasks = tl.load(num_tasks_per_wq + sm_id)
offset = INT_PER_TASK * NUM_SMS

for i in range(num_tasks):
{prof_load_begin}\
    task_type = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TASK_TYPE_OFFSET).to(tl.int32)
    layer_id = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + LAYER_ID_OFFSET).to(tl.int32)
    task_id = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TASK_ID_OFFSET).to(tl.int32)
    tile_id_or_start = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TILE_ID_OR_START_OFFSET).to(tl.int32)
    depend_entry_start = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + DEPEND_ENTRY_START_OFFSET).to(tl.int32)
    depend_entry_end = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + DEPEND_ENTRY_END_OFFSET).to(tl.int32)
    io_tensors_ptr = work_queues + i * offset + sm_id * INT_PER_TASK + IO_TENSORS_OFFSET

{prof_load_end}\
{prof_wait_begin}\
    with al.scope(core_mode="vector"):
        scoreboard_wait_deps_flat(
            scoreboard_ptr,
            task_deps_ptr,
            depend_entry_start,
            depend_entry_end,
            INT_PER_DEPS=INT_PER_DEPS,
            TILE_READY_SIGNAL=TILE_READY_SIGNAL,
            debug_counts=debug_counts,
        )

    # 访存保序
    if ({aic_cond}) and (depend_entry_end > depend_entry_start):
        with al.scope(core_mode="vector"):
            tl.sync_block_set('vector', 'cube', 5)
        with al.scope(core_mode="cube"):
            tl.sync_block_wait('vector', 'cube', 5)
    else:
        with al.scope(core_mode="vector"):
            dummy = tl.arange(0, 1)
            tl.inline_asm_elementwise(
                asm="BAR.ALL",
                constraints="=l,0",
                args=[dummy],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )

{prof_wait_end}\
    #### run task ####
{prof_task_begin}\
{textwrap.indent(tasks_dispatch_code.strip(), '    ')}
{prof_task_end}\
"""
    return body


def _make_npu_dynamic_scheduler_body(aic, enalbe_profiling, tasks_dispatch_code,
                                      scoreboard_wait_deps_task_type, load_before_wait_type,
                                      enable_task_prefetch):
    """生成 NPU 动态调度 (global flat queue + 原子竞争) 的 task 循环体. 所有行以 0 空格为基准缩进.

    参考 main 分支 GPU 动态调度设计:
    - 所有 task 放入一个全局平面队列 (enque_tasks 时 num_sms=1)
    - 每个 SM 通过 tl.atomic_add 原子抢下一个 task 索引, 返回旧值作为 task 编号
    - work_queue_start 是全局原子计数器 (单元素 int32 tensor)
    - 空闲 SM 自动承接更多工作, 消除静态预分配导致的负载不均

    Ascend 适配:
    - 每 SM 在循环外调一次 atomic_add 抢 chunk, 循环内零原子操作
    - DEMO 13/14 已验证: 8/8 SM 参与, 分布均匀, missing=0
    """
    # Triton 不支持 flat chained or, 必须嵌套括号: (A or (B or (C or D)))
    _parts = [f'(task_type=={x})' for x in aic]
    aic_cond = _parts[0]
    for p in _parts[1:]:
        aic_cond = f'({aic_cond} or {p})'

    # prof 字符串: 基准缩进 4 空格 (for 循环体层级), 外层 dedent+indent 后变为 8 空格
    prof_load_begin = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=1, task_type={load_before_wait_type}, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_load_end   = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=0, task_type={load_before_wait_type}, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_wait_begin = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=1, task_type={scoreboard_wait_deps_task_type}, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_wait_end   = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=0, task_type={scoreboard_wait_deps_task_type}, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_task_begin = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=1, task_type=task_type, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''
    prof_task_end   = f'    prof_offset = record_event(prof_base_ptr, prof_offset, prof_stride, sm_id, 0, 1, is_start=0, task_type=task_type, ENABLE_PROFILING=True)\n' if enalbe_profiling else ''

    # 从全局平面队列加载单条 task 元数据 + 等待依赖 + 执行 + 释放 (0-base 缩进)
    task_exec_block = f"""\
{prof_load_begin}\
task_type = tl.load(work_queues + cur_task_idx * INT_PER_TASK + TASK_TYPE_OFFSET).to(tl.int32)
layer_id = tl.load(work_queues + cur_task_idx * INT_PER_TASK + LAYER_ID_OFFSET).to(tl.int32)
task_id = tl.load(work_queues + cur_task_idx * INT_PER_TASK + TASK_ID_OFFSET).to(tl.int32)
tile_id_or_start = tl.load(work_queues + cur_task_idx * INT_PER_TASK + TILE_ID_OR_START_OFFSET).to(tl.int32)
depend_entry_start = tl.load(work_queues + cur_task_idx * INT_PER_TASK + DEPEND_ENTRY_START_OFFSET).to(tl.int32)
depend_entry_end = tl.load(work_queues + cur_task_idx * INT_PER_TASK + DEPEND_ENTRY_END_OFFSET).to(tl.int32)
io_tensors_ptr = work_queues + cur_task_idx * INT_PER_TASK + IO_TENSORS_OFFSET

{prof_load_end}\
{prof_wait_begin}\
with al.scope(core_mode="vector"):
    scoreboard_wait_deps_flat(
        scoreboard_ptr,
        task_deps_ptr,
        depend_entry_start,
        depend_entry_end,
        INT_PER_DEPS=INT_PER_DEPS,
        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
        debug_counts=debug_counts,
    )

# 访存保序
if ({aic_cond}) and (depend_entry_end > depend_entry_start):
    with al.scope(core_mode="vector"):
        tl.sync_block_set('vector', 'cube', 5)
    with al.scope(core_mode="cube"):
        tl.sync_block_wait('vector', 'cube', 5)
else:
    with al.scope(core_mode="vector"):
        dummy = tl.arange(0, 1)
        tl.inline_asm_elementwise(
            asm="BAR.ALL",
            constraints="=l,0",
            args=[dummy],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

{prof_wait_end}\
#### run task ####
{prof_task_begin}\
{textwrap.indent(tasks_dispatch_code.strip(), '    ')}
{prof_task_end}\
"""

    # chunk 调度: atomic_add 抢 chunk, for-range 用范围 0..chunk_size (和静态
    # 调度一样的 range(end) 结构), 内部加 my_start 偏移.
    # 注意: range(my_start, my_end) 带双参数在 Triton IR 中生成 while-like 结构,
    # 会触发 Bisheng SIGSEGV; 必须用 range(chunk_size) + 内部偏移.
    body = f"""\
num_total_tasks = tl.load(num_tasks_per_wq)
chunk_size = (num_total_tasks + NUM_SMS - 1) // NUM_SMS
with al.scope(core_mode="vector"):
    my_start = tl.atomic_add(work_queue_start, chunk_size)
if my_start < num_total_tasks:
    for i in range(chunk_size):
        cur_task_idx = my_start + i
        if cur_task_idx < num_total_tasks:
{textwrap.indent(textwrap.dedent(task_exec_block), '        ')}
"""
    return body


class CodeGenerator:

    def __init__(self):
        self._condition_and_codes: Dict[int, List[Tuple[CodeGenKey, str]]] = {}
        self._variable_names: Dict[str, str] = {
            "layer_id": "task_base_info.layer_id",
            "task_id": "task_base_info.task_id",
            "task_type": "task_type",
        }
        self._task_types_and_str: Dict[int, str] = {}

    def generate_task_dispatch_code(self, condition: CodeGenKey, code: str, is_first_branch=True) -> str:
        if condition.only_use_task_type():
            return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["task_type"]} == {condition.task_type}: # {self._task_types_and_str[condition.task_type]}
{textwrap.indent(code.strip(), '    ')}
"""
        else:
            return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["task_type"]} == {condition.task_type}: # {self._task_types_and_str[condition.task_type]}
    if {self._variable_names["layer_id"]} == {condition.layer_id} and {self._variable_names["task_id"]} == {condition.task_id}:
{textwrap.indent(code.strip(), '        ')}
"""

    def generate_for_each_task(self, condition: CodeGenKey, code: str, is_first_branch=True):
        return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["layer_id"]} == {condition.layer_id} and {self._variable_names["task_id"]} == {condition.task_id}:
{textwrap.indent(code.strip(), '    ')}
"""

    def generate_for_each_task_type(self, key_and_tasks_list, is_first_branch=True) -> str:
        assert len(key_and_tasks_list) > 0
        same_code = True
        task_type = key_and_tasks_list[0][0].task_type
        for key, code in key_and_tasks_list:
            if code != key_and_tasks_list[0][1]:
                same_code = False
        if same_code:
            code = key_and_tasks_list[0][1]
            return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["task_type"]} == {task_type}: # {self._task_types_and_str[task_type]}
{textwrap.indent(code.strip(), '    ')}
"""
        else:
            # each op may split into multi task, these tasks have same (task_type, layer_id, task_id)
            # only need to generate code once for these tasks
            already_generated = set()
            all_codes = ""
            is_first_task = True
            for key, code in key_and_tasks_list:
                if key in already_generated:
                    continue
                already_generated.add(key)
                cur_code = self.generate_for_each_task(key, code, is_first_task)
                all_codes += cur_code
                is_first_task = False
            return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["task_type"]} == {task_type}: # {self._task_types_and_str[task_type]}
{textwrap.indent(all_codes.strip(), '    ')}
"""

    def generate_code(self, tasks: List['TaskBase'], codegen_options: CodeGenOptions) -> str:
        self._condition_and_codes.clear()
        self._task_types_and_str.clear()

        target_hw = codegen_options.target_hw
        for task in tasks:
            key = task.get_codegen_key(task.layer_id, task.task_id)
            assert isinstance(key, CodeGenKey)
            task_type = type(task)
            code = registry.get_codegen(task_type)(task, target_hw)
            if key.task_type not in self._condition_and_codes:
                self._condition_and_codes[key.task_type] = []
            self._condition_and_codes[key.task_type].append((key, code))
            self._task_types_and_str[key.task_type] = task_type.__name__

        # TODO(zhengxuegui.0): branch optimization
        is_first_branch = True
        tasks_dispatch_code = ""

        for task_type, key_and_tasks_list in self._condition_and_codes.items():
            tasks_dispatch_code += self.generate_for_each_task_type(key_and_tasks_list, is_first_branch)
            is_first_branch = False

        mege_kernel_src, self._task_types_and_str = make_mega_kernel_src(tasks_dispatch_code, codegen_options,
                                                                         self._task_types_and_str)
        return mege_kernel_src, self._task_types_and_str