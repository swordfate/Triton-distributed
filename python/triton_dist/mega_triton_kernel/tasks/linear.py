import torch
from typing import Any, Dict, List
from .utils import cdiv
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency, InputDependencyDesc, OutputTilingDesc, DeviceProp
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class LinearConfig(ConfigBase):
    BLOCK_SIZE_M: int = 16
    BLOCK_SIZE_N: int = 128
    SUB_BLOCK_SIZE_N: int = 128
    BLOCK_SIZE_K: int = 128
    NUM_STAGES: int = 4


@dataclass
class LinearTask(TaskBase):
    config: LinearConfig


@dataclass
class MLPFC1Config(LinearConfig):
    pass


@dataclass
class MLPFC1Task(LinearTask):
    config: MLPFC1Config


@dataclass
class MLPFC2Config(LinearConfig):
    pass


@dataclass
class MLPFC2Task(LinearTask):
    config: MLPFC2Config


@dataclass
class QKVProjTask(LinearTask):
    config: LinearConfig


@dataclass
class OProjTask(LinearTask):
    config: LinearConfig


def linear_config_factory(**kwargs) -> LinearConfig:
    return dataclasses.replace(LinearConfig(), **kwargs)


def mlp_fc1_config_factory(**kwargs) -> MLPFC1Config:
    default = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 64,
        'SUB_BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 128,
        'NUM_STAGES': 6,
    }
    default.update(kwargs)
    return MLPFC1Config(**default)


def mlp_fc2_config_factory(**kwargs) -> MLPFC2Config:
    default = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 64,
        'SUB_BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 256,
        'NUM_STAGES': 6,
    }
    default.update(kwargs)
    return MLPFC2Config(**default)


def _make_npu_fc1_inline_code(BLOCK_SIZE_M, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES):
    """展开 fc1_task_compute 的 task_base_info_get_tensor, 其余调用不动. (sync id=11, K align=16)"""
    return f"""
IPT = MAX_NUM_TENSOR_DIMS + 2
input = io_tensors_ptr + 0 * IPT
weight = io_tensors_ptr + 1 * IPT
output = io_tensors_ptr + 2 * IPT
M = tensor_desc_size(input, 0)
K = tensor_desc_size(input, 1, 16)
N = tensor_desc_size(weight, 0)
a_ptr = tensor_desc_data_ptr(input, tl.bfloat16)
b_ptr = tensor_desc_data_ptr(weight, tl.bfloat16)
c_ptr = tensor_desc_data_ptr(output, tl.bfloat16)
tile_id = tile_id_or_start
tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K,
    {BLOCK_SIZE_M}, {BLOCK_SIZE_N}, {SUB_BLOCK_SIZE_N}, {BLOCK_SIZE_K}, {NUM_STAGES})
with al.scope(core_mode="cube"):
    tl.sync_block_set('cube', 'vector', 11)
with al.scope(core_mode="vector"):
    tl.sync_block_wait('cube', 'vector', 11)
scoreboard_release_tile_flat(
    scoreboard_ptr=scoreboard_ptr,
    layer_id=layer_id,
    task_id=task_id,
    tile_id=tile_id_or_start,
    TILE_READY_SIGNAL=TILE_READY_SIGNAL,
    MAX_TASK_ID=MAX_TASK_ID,
    MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
)
"""


def _make_npu_linear_task_inline_code(BLOCK_SIZE_M, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES, ALIGNMENT_K):
    """展开 linear_task_compute, 保留 ALIGNMENT_K 参数. (sync id=10)"""
    return f"""
IPT = MAX_NUM_TENSOR_DIMS + 2
input = io_tensors_ptr + 0 * IPT
weight = io_tensors_ptr + 1 * IPT
output = io_tensors_ptr + 2 * IPT
M = tensor_desc_size(input, 0)
K = tensor_desc_size(input, 1, {ALIGNMENT_K})
N = tensor_desc_size(weight, 0)
a_ptr = tensor_desc_data_ptr(input, tl.bfloat16)
b_ptr = tensor_desc_data_ptr(weight, tl.bfloat16)
c_ptr = tensor_desc_data_ptr(output, tl.bfloat16)
tile_id = tile_id_or_start
tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K,
    {BLOCK_SIZE_M}, {BLOCK_SIZE_N}, {SUB_BLOCK_SIZE_N}, {BLOCK_SIZE_K}, {NUM_STAGES})
with al.scope(core_mode="cube"):
    tl.sync_block_set('cube', 'vector', 10)
with al.scope(core_mode="vector"):
    tl.sync_block_wait('cube', 'vector', 10)
scoreboard_release_tile_flat(
    scoreboard_ptr=scoreboard_ptr,
    layer_id=layer_id,
    task_id=task_id,
    tile_id=tile_id_or_start,
    TILE_READY_SIGNAL=TILE_READY_SIGNAL,
    MAX_TASK_ID=MAX_TASK_ID,
    MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
)
"""


def codegen_linear(task: LinearTask, target_hw: str) -> str:
    config: MLPFC1Config = task.config
    a, b = task.io_tensors[0]
    M, K = a.shape
    ALIGNMENT_K = 1
    if K % 16 == 0:
        ALIGNMENT_K = 16
    if target_hw == 'gpu':
        code = f"""
linear_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES}, ALIGNMENT_K={ALIGNMENT_K})
"""
    else:
        code = _make_npu_linear_task_inline_code(
            BLOCK_SIZE_M=config.BLOCK_SIZE_M, BLOCK_SIZE_N=config.BLOCK_SIZE_N,
            SUB_BLOCK_SIZE_N=config.SUB_BLOCK_SIZE_N, BLOCK_SIZE_K=config.BLOCK_SIZE_K,
            NUM_STAGES=config.NUM_STAGES, ALIGNMENT_K=ALIGNMENT_K)
    return code


def codegen_mlp_fc1(task: MLPFC1Task, target_hw: str) -> str:
    config: MLPFC1Config = task.config
    if target_hw == 'gpu':
        code = f"""
fc1_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES})
"""
    else:
        code = _make_npu_fc1_inline_code(
            BLOCK_SIZE_M=config.BLOCK_SIZE_M, BLOCK_SIZE_N=config.BLOCK_SIZE_N,
            SUB_BLOCK_SIZE_N=config.SUB_BLOCK_SIZE_N, BLOCK_SIZE_K=config.BLOCK_SIZE_K,
            NUM_STAGES=config.NUM_STAGES)
    return code


def codegen_mlp_fc2(task: MLPFC2Task, target_hw: str) -> str:
    config: MLPFC2Config = task.config
    if target_hw == 'gpu':
        code = f"""
fc1_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES})
"""
    else:
        code = _make_npu_fc1_inline_code(
            BLOCK_SIZE_M=config.BLOCK_SIZE_M, BLOCK_SIZE_N=config.BLOCK_SIZE_N,
            SUB_BLOCK_SIZE_N=config.SUB_BLOCK_SIZE_N, BLOCK_SIZE_K=config.BLOCK_SIZE_K,
            NUM_STAGES=config.NUM_STAGES)
    return code


def codegen_qkv_proj(task: QKVProjTask, target_hw: str) -> str:
    config: LinearConfig = task.config
    if target_hw == 'gpu':
        code = f"""
fc1_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES})
"""
    else:
        code = _make_npu_fc1_inline_code(
            BLOCK_SIZE_M=config.BLOCK_SIZE_M, BLOCK_SIZE_N=config.BLOCK_SIZE_N,
            SUB_BLOCK_SIZE_N=config.SUB_BLOCK_SIZE_N, BLOCK_SIZE_K=config.BLOCK_SIZE_K,
            NUM_STAGES=config.NUM_STAGES)
    return code


def codegen_o_proj(task: OProjTask, target_hw: str) -> str:
    config: LinearConfig = task.config
    if target_hw == 'gpu':
        code = f"""
fc1_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES})
"""
    else:
        code = _make_npu_fc1_inline_code(
            BLOCK_SIZE_M=config.BLOCK_SIZE_M, BLOCK_SIZE_N=config.BLOCK_SIZE_N,
            SUB_BLOCK_SIZE_N=config.SUB_BLOCK_SIZE_N, BLOCK_SIZE_K=config.BLOCK_SIZE_K,
            NUM_STAGES=config.NUM_STAGES)
    return code


# @registry.register_task(op_type="linear", task_cls=LinearTask, config_factory=linear_config_factory,
#                         codegen_func=codegen_linear)
class LinearTaskBaseBuilder(TaskBuilderBase):

    @classmethod
    def get_problem_size(cls, io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any], target_hw='gpu'):
        a, b = io_tensors[0]
        M, K = a.shape
        N, K = b.shape
        return (M, N, K)

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True, config_args={}, target_hw='gpu') -> List[TaskBase]:
        assert tile_wise == True  # noqa: E712
        kernel_config = cls.create_config(**config_args)
        task_id = cls.get_task_id(layer_id)
        M, N, K = cls.get_problem_size(io_tensors, extra_params)
        num_sm = device_prop.NUM_SMS
        # BLOCK_SIZE_M = kernel_config.BLOCK_SIZE_M
        # BLOCK_SIZE_N = kernel_config.BLOCK_SIZE_N
        BLOCK_SIZE_M = 16
        SUB_BLOCK_SIZE_N = 320
        BLOCK_SIZE_K = 320
        num_tiles_m = cdiv(M, BLOCK_SIZE_M)
        
        best_i = 0
        max_use = 0
        for i in range(26):
            SUB_BLOCK_SIZE_N = 416 - i * 16
            num_sub_tiles_n = cdiv(N, SUB_BLOCK_SIZE_N)
            num_sub_tiles = num_tiles_m * num_sub_tiles_n
            left = (num_sub_tiles + num_sm - 1) % num_sm
            if left > max_use:
                best_i = i
                max_use = left

        SUB_BLOCK_SIZE_N = 416 - best_i * 16
        best_i = 416 // (SUB_BLOCK_SIZE_N // 16)
        if best_i >= 16:
            best_i = best_i // 16 * 16
        BLOCK_SIZE_K = best_i * 16

        num_sub_tiles_n = cdiv(N, SUB_BLOCK_SIZE_N)
        num_sub_tiles = num_tiles_m * num_sub_tiles_n
        sub_blocks = cdiv(num_sub_tiles, num_sm)

        BLOCK_SIZE_N = SUB_BLOCK_SIZE_N * sub_blocks
        num_tiles_n = cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_tiles_m * num_tiles_n

        if N == 151936:
            BLOCK_SIZE_M = 16
            BLOCK_SIZE_N = 7728
            BLOCK_SIZE_K = 256
            SUB_BLOCK_SIZE_N = 336

        kernel_config.BLOCK_SIZE_M = BLOCK_SIZE_M
        kernel_config.BLOCK_SIZE_N = BLOCK_SIZE_N
        kernel_config.BLOCK_SIZE_K = BLOCK_SIZE_K
        kernel_config.SUB_BLOCK_SIZE_N = SUB_BLOCK_SIZE_N

        # print(f"Linear Tast of N:{N}, M:{M}, K:{K}, BLOCK_N:{BLOCK_SIZE_N}, SUB_BLOCK_N:{SUB_BLOCK_SIZE_N}, BLOCK_K:{BLOCK_SIZE_K}, BLOCK_M:{BLOCK_SIZE_M}")
        # print(f"BLOCKS N:{num_tiles_n}, BLOCKS M:{num_tiles_m}, BLOCKS K:{cdiv(K, BLOCK_SIZE_K)}, SUB BLOCKS:{sub_blocks}")

        x, w = io_tensors[0]
        y = io_tensors[1][0]

        tasks = []
        # cls.log(
        #     f"Linear Task: M = {M}, N = {N}, K = {K}, num_tiles = {num_tiles}, num_sm = {num_sm}, tile_wise = {tile_wise}, dependency = {dependency}, BLOCK_SIZE_M ={BLOCK_SIZE_M}, BLOCK_SIZE_N = {BLOCK_SIZE_N}, task_id = {task_id}"
        # )
        print(f"Linear Task: M = {M}, N = {N}, K = {K}, num_tiles = {num_tiles}, num_sm = {num_sm}, tile_wise = {tile_wise}, dependency = {dependency}, BLOCK_SIZE_M ={BLOCK_SIZE_M}, BLOCK_SIZE_N = {BLOCK_SIZE_N}, task_id = {task_id}")
        for tm in range(num_tiles_m):
            for tn in range(num_tiles_n):
                tile_id = tm * num_tiles_n + tn
                bm = min(BLOCK_SIZE_M, M - tm * BLOCK_SIZE_M)
                bn = min(BLOCK_SIZE_N, N - tn * BLOCK_SIZE_N)
                x_desc = InputDependencyDesc(x, require_full=False, start_indices=(tm * BLOCK_SIZE_M, 0),
                                             data_sizes=(bm, K))
                w_desc = InputDependencyDesc(w, require_full=False, start_indices=(tn * BLOCK_SIZE_N, 0),
                                             data_sizes=(bn, K))
                y_desc = OutputTilingDesc(tile_sizes=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                                          start_indices=(tm * BLOCK_SIZE_M, tn * BLOCK_SIZE_N))
                inputs_dep = {x: x_desc, w: w_desc}
                outs_tile_mapping = {y: y_desc}
                tasks.append(
                    cls._create_task(layer_id, task_id, tile_id, num_tiles, kernel_config, dependency, io_tensors,
                                     extra_params, inputs_dep, outs_tile_mapping))
        return tasks

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any], target_hw='gpu') -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params)

@registry.register_task(op_type="linear", task_cls=LinearTask, config_factory=linear_config_factory,
                        codegen_func=codegen_linear)
class LinearTaskBuilder(LinearTaskBaseBuilder):

    @classmethod

    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any], target_hw='gpu') -> List[TaskBase]:
        if target_hw == 'npu':
            config_args = {
                "BLOCK_SIZE_N": 7728,
                "SUB_BLOCK_SIZE_N": 336,
                "BLOCK_SIZE_K": 256,
            }
        else:
            config_args = {}
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, target_hw=target_hw, config_args=config_args)

@registry.register_task(op_type="mlp_fc1", task_cls=MLPFC1Task, config_factory=mlp_fc1_config_factory,
                        codegen_func=codegen_linear)
class MLPFC1TaskBuilder(LinearTaskBaseBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any], target_hw='gpu') -> List[TaskBase]:
        if target_hw == 'npu':
            config_args = {
                "BLOCK_SIZE_N": 1248,
                "SUB_BLOCK_SIZE_N":208,
                "BLOCK_SIZE_K": 512,
            }
        else:
            config_args = {}
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True, 
                                    target_hw=target_hw, config_args=config_args)


# reduce branch in mega kernel, just use task type as condition
@registry.register_task(op_type="mlp_fc2", task_cls=MLPFC2Task, config_factory=mlp_fc2_config_factory,
                        codegen_func=codegen_linear)
class MLPFC2TaskBuilder(LinearTaskBaseBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any], target_hw='gpu') -> List[TaskBase]:
        if target_hw == 'npu':
            config_args = {
                "BLOCK_SIZE_N": 208,
                "SUB_BLOCK_SIZE_N":208,
                "BLOCK_SIZE_K": 512,
            }
        else:
            config_args = {}
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True, 
                                    target_hw=target_hw, config_args=config_args)


@registry.register_task(op_type="qkv_proj", task_cls=QKVProjTask, config_factory=linear_config_factory,
                        codegen_func=codegen_linear)
class QKVProjTaskBuilder(LinearTaskBaseBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any], target_hw='gpu') -> List[TaskBase]:
        if target_hw == 'npu':
            config_args = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 320,
                "SUB_BLOCK_SIZE_N":320,
                "BLOCK_SIZE_K": 256,
                "NUM_STAGES": 5,
            }
        else:
            config_args = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 256,
                "NUM_STAGES": 5,
            }
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True,
                                     config_args=config_args, target_hw=target_hw)


@registry.register_task(op_type="o_proj", task_cls=OProjTask, config_factory=linear_config_factory,
                        codegen_func=codegen_linear)
class OProjTaskBuilder(LinearTaskBaseBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any], target_hw='gpu') -> List[TaskBase]:
        if target_hw == 'npu':
            config_args = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 208,
                "SUB_BLOCK_SIZE_N":208,
                "BLOCK_SIZE_K": 512,
                "NUM_STAGES": 7,
            }
        else:
            config_args = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "NUM_STAGES": 7,
            }
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True,
                                     config_args=config_args, target_hw=target_hw)