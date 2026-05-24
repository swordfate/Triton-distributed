from typing import Tuple, List
from .utils import cdiv
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency, InputDependencyDesc, OutputTilingDesc
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase

import acl


@dataclass
class AllReduceConfig(ConfigBase):
    BLOCK_SIZE: int = 1024


@dataclass
class AllReduceTask(TaskBase):
    config: AllReduceConfig



def allreduce_config_factory(**kwargs) -> AllReduceConfig:
    return dataclasses.replace(AllReduceConfig(), **kwargs)


def codegen_allreduce(task: AllReduceConfig) -> str:
    config: AllReduceConfig = task.config

    code = f"""
allreduce_task_compute(task_base_info, scoreboard, BLOCK_SIZE={config.BLOCK_SIZE})
"""
    return code


@dataclass
class AscendAllReduceConfig(ConfigBase):
    BLOCK_SIZE_B: int = 1
    BLOCK_SIZE_H: int = 208


@dataclass
class AscendAllReduceTask(TaskBase):
    config: AscendAllReduceConfig
    def extra_params_to_tuple(self) -> Tuple[int]:
        return (self.extra_params["rank"], self.extra_params["world_size"])

@dataclass
class GatherAllConfig(ConfigBase):
    BLOCK_SIZE_B: int = 1
    BLOCK_SIZE_H: int = 2048


@dataclass
class GatherAllTask(TaskBase):
    config: GatherAllConfig
    def extra_params_to_tuple(self) -> Tuple[int]:
        return (self.extra_params["rank"], self.extra_params["world_size"])


def ascend_allreduce_config_factory(**kwargs) -> AscendAllReduceConfig:
    return dataclasses.replace(AscendAllReduceConfig(), **kwargs)

def gather_all_config_factory(**kwargs) -> GatherAllConfig:
    return dataclasses.replace(GatherAllConfig(), **kwargs)

def codegen_allreduce_ascend(task: AscendAllReduceConfig, target_hw: str) -> str:
    config: AscendAllReduceTask = task.config 

    code = f"""
with al.scope(core_mode="vector"):
    allreduce_task_compute(
        tile_id_or_start=tile_id_or_start, MAX_NUM_TENSOR_DIMS=MAX_NUM_TENSOR_DIMS,
        io_tensors_ptr=io_tensors_ptr,
        BLOCK_SIZE_B={config.BLOCK_SIZE_B},
        BLOCK_SIZE_H={config.BLOCK_SIZE_H},
        scoreboard_ptr=scoreboard_ptr,
        layer_id=layer_id,
        task_id=task_id,
        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
        MAX_TASK_ID=MAX_TASK_ID,
        MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
    )
"""
    return code

def codegen_gather_all(task: GatherAllConfig, target_hw: str) -> str:
    config: GatherAllTask = task.config 

    code = f"""
with al.scope(core_mode="vector"):
    gather_all_compute(
        tile_id_or_start=tile_id_or_start, MAX_NUM_TENSOR_DIMS=MAX_NUM_TENSOR_DIMS,
        io_tensors_ptr=io_tensors_ptr,
        BLOCK_SIZE_B={config.BLOCK_SIZE_B},
        BLOCK_SIZE_H={config.BLOCK_SIZE_H},
        scoreboard_ptr=scoreboard_ptr,
        layer_id=layer_id,
        task_id=task_id,
        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
        MAX_TASK_ID=MAX_TASK_ID,
        MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
    )
"""
    return code


@registry.register_task(op_type="allreduce", task_cls=AllReduceTask, config_factory=allreduce_config_factory,
                        codegen_func=codegen_allreduce)
class AllReduceTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        input, output = io_tensors[0][0], io_tensors[1][0]
        num_elements = output.numel()
        # assert input.shape == output.shape
        assert len(input.shape) == 1
        # assert len(output.shape) == 1
        assert num_elements * output.element_size() % 128 == 0
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        num_tiles = cdiv(num_elements, kernel_config.BLOCK_SIZE)

        cls.log(
            f"AllReduce Task: num_tiles = {num_tiles}, num_elements = {num_elements}, BLOCK_SIZE = {kernel_config.BLOCK_SIZE}, dependency = {dependency}"
        )
        tasks = []
        for i in range(num_tiles):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params))
        return tasks

@registry.register_task(op_type="allreduce_ascend", task_cls=AscendAllReduceTask, config_factory=ascend_allreduce_config_factory,
                        codegen_func=codegen_allreduce_ascend)
class AscendAllReduceTaskBuilder(TaskBuilderBase):
    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True, target_hw='gpu'):
        # 这里的 io_tensors[0] = [input, barrier_global, barrier_intra]
        # 这里的 io_tensors[1] = [output]
        input = io_tensors[0][0]
        output = io_tensors[1][0]
        B, H = output.shape
        num_elements = output.numel()
        # assert input.shape == output.shape 
        # assert len(input.shape) == 1 and len(output.shape) == 1
        assert num_elements * output.element_size() % 128 == 0
        
        # 按ncores切分任务
        num_tiles = 20 if acl.get_soc_name()=='Ascend910B3' else 24
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        kernel_config.BLOCK_SIZE_B = (B + extra_params['world_size'] - 1) // extra_params['world_size']
        
        tasks = []
        for i in range(num_tiles):
            # input_desc = InputDependencyDesc(input, require_full=False, start_indices=(0,), data_sizes=(output.shape[0],)) # 只需前半段
            # out_desc = OutputTilingDesc(start_indices=(0,), tile_sizes=(output.shape[0],))
            # inputs_dep = {input: input_desc}
            # outs_tile_mapping = {output: out_desc}
            tasks.append(cls._create_task(layer_id, task_id, i, num_tiles, 
                    kernel_config, dependency, io_tensors, extra_params))
        return tasks

@registry.register_task(op_type="gather_all", task_cls=GatherAllTask, config_factory=gather_all_config_factory,
                        codegen_func=codegen_gather_all)
class GatherAllTaskBuilder(TaskBuilderBase):
    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True, target_hw='gpu'):
        input = io_tensors[0][0]
        output = io_tensors[1][0]
        num_elements = output.numel()
        B, H = output.shape
        # assert len(input.shape) == 1 and len(output.shape) == 1
        assert num_elements * output.element_size() % 128 == 0
        
        # 按ncores切分任务
        num_tiles = 20 if acl.get_soc_name()=='Ascend910B3' else 24
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        kernel_config.BLOCK_SIZE_B = B
        kernel_config.BLOCK_SIZE_H = kernel_config.BLOCK_SIZE_H // B

        tasks = []
        for i in range(num_tiles):
            tasks.append(cls._create_task(layer_id, task_id, i, num_tiles, 
                    kernel_config, dependency, io_tensors, extra_params))
        return tasks