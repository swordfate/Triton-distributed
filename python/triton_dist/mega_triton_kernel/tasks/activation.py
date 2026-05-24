from typing import Any, Dict, List
from .utils import cdiv
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency, InputDependencyDesc, OutputTilingDesc, DeviceProp
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase
import torch


@dataclass
class SiLUMulUpConfig(ConfigBase):
    BLOCK_SIZE_M: int = 8
    BLOCK_SIZE_N: int = 128


@dataclass
class SiLUMulUpTask(TaskBase):
    config: SiLUMulUpConfig


def silu_mul_up_config_factory(**kwargs) -> SiLUMulUpConfig:
    default = {
        'BLOCK_SIZE_M': 8,
        'BLOCK_SIZE_N': 128,
    }
    default.update(kwargs)
    return SiLUMulUpConfig(**default)


def codegen_silu_mul_up_fc1(task: SiLUMulUpTask, target_hw: str) -> str:
    config: SiLUMulUpConfig = task.config
    if target_hw == 'gpu':
        code = f"""
silu_mul_up_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N})
"""
    else:
        code = f"""
with al.scope(core_mode="vector"):
    silu_mul_up_task_compute(
                            tile_id_or_start=tile_id_or_start, MAX_NUM_TENSOR_DIMS=MAX_NUM_TENSOR_DIMS,
                            io_tensors_ptr=io_tensors_ptr,
                            BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                            scoreboard_ptr=scoreboard_ptr,
                            layer_id=layer_id,
                            task_id=task_id,
                            TILE_READY_SIGNAL=TILE_READY_SIGNAL,
                            MAX_TASK_ID=MAX_TASK_ID,
                            MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP,
                            )
"""
    return code


@registry.register_task(op_type="silu_mul_up", task_cls=SiLUMulUpTask, config_factory=silu_mul_up_config_factory,
                        codegen_func=codegen_silu_mul_up_fc1)
class SiLUMulUpTaskBuilder(TaskBuilderBase):

    @classmethod
    def get_problem_size(cls, io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]):
        output = io_tensors[1][0]
        M, N = output.shape
        return (M, N)

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True, target_hw='gpu') -> List[TaskBase]:
        kernel_config = cls.create_config()
        if target_hw == 'npu':
            kernel_config.BLOCK_SIZE_N = 1280
        task_id = cls.get_task_id(layer_id)
        BLOCK_SIZE_M = kernel_config.BLOCK_SIZE_M
        BLOCK_SIZE_N = kernel_config.BLOCK_SIZE_N
        x, y = io_tensors[0][0], io_tensors[1][0]
        M, N = cls.get_problem_size(io_tensors, extra_params)
        num_tiles_m = cdiv(M, BLOCK_SIZE_M)
        num_tiles_n = cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_tiles_m * num_tiles_n
        num_sm = device_prop.NUM_SMS
        tasks = []
        cls.log(
            f"SiLUMulUp Task: M = {M}, N = {N}, num_tiles = {num_tiles}, num_sm = {num_sm}, tile_wise = {tile_wise}, dependency = {dependency}")
        for tm in range(num_tiles_m):
            for tn in range(num_tiles_n):
                tile_id = tm * num_tiles_n + tn
                bm = min(BLOCK_SIZE_M, M - tm * BLOCK_SIZE_M)
                bn = min(BLOCK_SIZE_N, N - tn * BLOCK_SIZE_N) + N
                x_desc = InputDependencyDesc(x, require_full=False,
                                             start_indices=(tm * BLOCK_SIZE_M, tn * BLOCK_SIZE_N), data_sizes=(bm, bn))
                y_desc = OutputTilingDesc(tile_sizes=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                                          start_indices=(tm * BLOCK_SIZE_M, tn * BLOCK_SIZE_N))
                inputs_dep = {
                    x: x_desc,
                }
                outs_tile_mapping = {y: y_desc}
                tasks.append(
                    cls._create_task(layer_id, task_id, tile_id, num_tiles, kernel_config, dependency, io_tensors,
                                     extra_params, inputs_dep, outs_tile_mapping))
        return tasks

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any], target_hw: str) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, target_hw=target_hw)