from triton import next_power_of_2
from typing import Tuple, List
import dataclasses
from dataclasses import dataclass
from .utils import build_tile_desc, torch_dtype_to_triton_dtype_str, cdiv
from ..core.task_base import TaskBase, TaskDependency, InputDependencyDesc, OutputTilingDesc
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class QKVPackQKNormRopeSplitVConfig(ConfigBase):
    BLOCK_SEQ: int = 128
    BLOCK_HD: int = 256


@dataclass
class QKNormRopeUpdateKVCacheConfig(ConfigBase):
    BLOCK_SIZE_B: int = 1

@dataclass
class RMSNormConfig(ConfigBase):
    BLOCK_SIZE_N: int = 4096


@dataclass
class QKNormRopeUpdateKVCacheTask(TaskBase):
    config: QKNormRopeUpdateKVCacheConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # rms_eps/rope_theta as constexpr in codegen
        return ()


@dataclass
class QKVPackQKNormRopeSplitVTask(TaskBase):
    config: QKVPackQKNormRopeSplitVConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # rms_eps as constexpr in codegen
        return ()


@dataclass
class RMSNormTask(TaskBase):
    config: RMSNormConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # rms_eps as constexpr in codegen
        return ()


def qk_norm_rope_update_kvcache_config_factory(**kwargs) -> QKNormRopeUpdateKVCacheConfig:
    return dataclasses.replace(QKNormRopeUpdateKVCacheConfig(), **kwargs)


def qkv_pack_qk_norm_rope_split_v_config_factory(**kwargs) -> QKVPackQKNormRopeSplitVConfig:
    return dataclasses.replace(QKVPackQKNormRopeSplitVConfig(), **kwargs)


def rms_norm_config_factory(**kwargs) -> RMSNormConfig:
    return dataclasses.replace(RMSNormConfig(), **kwargs)


def codegen_qk_norm_rope_update_kvcache(task: QKNormRopeUpdateKVCacheTask, target_hw: str) -> str:
    config : QKNormRopeUpdateKVCacheConfig = task.config
    qkv, block_tables, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = task.io_tensors[0]
    key_cache, value_cache, q_norm_rope = task.io_tensors[1]
    Q_HEAD_DIM = qkv.shape[-1]
    V_HEAD_DIM = value_cache.shape[-1]
    NUM_KV_HEADS = key_cache.shape[-2]
    NUM_Q_HEADS = qkv.shape[-2] - 2 * NUM_KV_HEADS
    PAGE_SIZE, NUM_KV_HEADS, V_HEAD_DIM = value_cache.shape[-3], value_cache.shape[-2], value_cache.shape[-1]
    MAX_NUM_BLOCKS_PER_SEQ = block_tables.shape[-1]
    if target_hw == 'gpu':
        code = f"""
rmsnorm_rope_update_kv_cache_task_compute(
    task_base_info, scoreboard, NUM_Q_HEADS={NUM_Q_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, Q_HEAD_DIM={Q_HEAD_DIM},
    V_HEAD_DIM={V_HEAD_DIM}, PAGE_SIZE={PAGE_SIZE}, MAX_NUM_BLOCKS_PER_SEQ={MAX_NUM_BLOCKS_PER_SEQ},
    Q_RMS_EPS={task.extra_params["q_rms_eps"]}, K_RMS_EPS={task.extra_params["k_rms_eps"]})
"""
    else:
        code = f"""
with al.scope(core_mode="vector"):
    rmsnorm_rope_update_kv_cache_task_compute(
                                            tile_id_or_start=tile_id_or_start, MAX_NUM_TENSOR_DIMS=MAX_NUM_TENSOR_DIMS,
                                            io_tensors_ptr=io_tensors_ptr,
                                            NUM_Q_HEADS={NUM_Q_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, Q_HEAD_DIM={Q_HEAD_DIM},
                                            V_HEAD_DIM={V_HEAD_DIM}, PAGE_SIZE={PAGE_SIZE}, MAX_NUM_BLOCKS_PER_SEQ={MAX_NUM_BLOCKS_PER_SEQ},
                                            Q_RMS_EPS={task.extra_params["q_rms_eps"]}, K_RMS_EPS={task.extra_params["k_rms_eps"]},
                                            BLOCK_SIZE_B={config.BLOCK_SIZE_B},
                                            scoreboard_ptr=scoreboard_ptr,
                                            layer_id=layer_id,
                                            task_id=task_id,
                                            TILE_READY_SIGNAL=TILE_READY_SIGNAL,
                                            MAX_TASK_ID=MAX_TASK_ID,
                                            MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP,
                                            )
"""
    return code


def codegen_rms_norm(task: RMSNormTask, target_hw: str) -> str:
    config: RMSNormConfig = task.config
    if target_hw == 'gpu':
        code = f"""
rmsnorm_task_compute(task_base_info, scoreboard, RMS_EPS={task.extra_params["rms_eps"]}, BLOCK_SIZE_N = {config.BLOCK_SIZE_N})
"""
    else:
        code = f"""
with al.scope(core_mode="vector"):
    rmsnorm_task_compute(
                        tile_id_or_start=tile_id_or_start, MAX_NUM_TENSOR_DIMS=MAX_NUM_TENSOR_DIMS,
                        io_tensors_ptr=io_tensors_ptr,
                        RMS_EPS={task.extra_params["rms_eps"]}, BLOCK_SIZE_N = {config.BLOCK_SIZE_N},
                        scoreboard_ptr=scoreboard_ptr,
                        layer_id=layer_id,
                        task_id=task_id,
                        TILE_READY_SIGNAL=TILE_READY_SIGNAL,
                        MAX_TASK_ID=MAX_TASK_ID,
                        MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP,
                        )
"""
    return code


def codegen_qkv_pack_qk_norm_rope_split_v(task: QKVPackQKNormRopeSplitVTask) -> str:
    qkv, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = task.io_tensors[0]
    q_norm_rope, k_norm_rope, v = task.io_tensors[1]
    HEAD_DIM = qkv.shape[-1]
    triton_dtype = torch_dtype_to_triton_dtype_str(qkv.dtype)
    NUM_Q_HEADS = q_norm_rope.shape[-2]
    NUM_KV_HEADS = k_norm_rope.shape[-2]
    code = f"""
qkv_pack_qk_norm_rope_split_v_task_compute(
    task_base_info, scoreboard, DTYPE={triton_dtype}, NUM_Q_HEADS={NUM_Q_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, HEAD_DIM={HEAD_DIM},
    Q_RMS_EPS={task.extra_params["q_rms_eps"]}, K_RMS_EPS={task.extra_params["k_rms_eps"]},
    BLOCK_SEQ={task.config.BLOCK_SEQ}, BLOCK_HD={task.config.BLOCK_HD},
)
"""
    return code


@registry.register_task(op_type="qk_norm_rope_update_kvcache", task_cls=QKNormRopeUpdateKVCacheTask,
                        config_factory=qk_norm_rope_update_kvcache_config_factory,
                        codegen_func=codegen_qk_norm_rope_update_kvcache)
class QKNormRopeUpdateKVCacheTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True, target_hw='gpu') -> List[TaskBase]:
        qkv, block_tables, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = io_tensors[0]
        key_cache, value_cache, q_norm_rope = io_tensors[1]
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        assert len(qkv.shape) == 4
        batch, seq_len, num_qkv_heads, head_dim = qkv.shape
        num_kv_heads = key_cache.shape[-2]
        num_qk_heads = num_qkv_heads - num_kv_heads
        num_q_heads = num_qkv_heads - 2 * num_kv_heads
        kernel_config.BLOCK_SIZE_B = cdiv(batch * seq_len * num_qk_heads, device_prop.NUM_SMS)
        kernel_config.BLOCK_SIZE_B = min(batch, kernel_config.BLOCK_SIZE_B)
        BLOCK_SIZE_B = kernel_config.BLOCK_SIZE_B
        block_b = cdiv(batch, BLOCK_SIZE_B)
        num_tiles = block_b * seq_len * num_qk_heads
        cls.log(f"KNormRopeUpdateKVCache Task: num_tiles = {num_tiles}, task_id = {task_id}, dependency = {dependency}")
        tasks = []
        for i in range(num_tiles):
            # tile → (batch_group, seq, head) 映射 (与 kernel 一致)
            idx_0 = i // num_qk_heads
            idx_1 = i % num_qk_heads
            batch_grp_idx = idx_0 // seq_len
            seq_idx = idx_0 % seq_len
            batch_start = batch_grp_idx * BLOCK_SIZE_B
            valid_batch = min(BLOCK_SIZE_B, batch - batch_start)

            # 输入 qkv: 每个 tile 处理 batch_group 范围内的所有 head
            # (V 数据在 K head tile 中非连续, 因此使用全 head 范围近似)
            qkv_desc = InputDependencyDesc(
                qkv, require_full=False,
                start_indices=(batch_start, seq_idx, 0, 0),
                data_sizes=(valid_batch, 1, num_qkv_heads, head_dim))
            inputs_dep = {qkv: qkv_desc}

            # 输出 q_norm_rope: Q head tile 产生 tile 级输出, K head tile 不产生
            if idx_1 < num_q_heads:
                out_desc = OutputTilingDesc(
                    start_indices=(batch_start, seq_idx, idx_1, 0),
                    tile_sizes=(valid_batch, 1, 1, head_dim))
                # key_cache / value_cache 输出位置由 block_table 运行时决定, 保持 full-dep
                outs_tile_mapping = {q_norm_rope: out_desc}
            else:
                # K head tile 不产生 q_norm_rope, key_cache/value_cache 保持 full-dep
                empty_desc = OutputTilingDesc(start_indices=(0, 0, 0, 0), tile_sizes=(0, 0, 0, 0))
                outs_tile_mapping = {q_norm_rope: empty_desc}

            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params,
                                 inputs_dep=inputs_dep, outs_tile_mapping=outs_tile_mapping))
        return tasks


@registry.register_task(op_type="rms_norm", task_cls=RMSNormTask, config_factory=rms_norm_config_factory,
                        codegen_func=codegen_rms_norm)
class RMSNormTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True, target_hw='gpu') -> List[TaskBase]:
        input, weight = io_tensors[0]
        output = io_tensors[1][0]
        num_tiles = output.numel() // output.shape[-1]
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        if target_hw == 'npu':
            kernel_config.BLOCK_SIZE_N = 4096
        # cls.log(f"RMS Norm Task: num_tiles = {num_tiles}, task_id = {task_id}, dependency = {dependency}")
        print(f"RMS Norm Task: num_tiles = {num_tiles}, task_id = {task_id}, dependency = {dependency}")
        tasks = []
        tile_size = output.shape[-1]
        for i in range(num_tiles):
            in_start_indices, in_data_sizes = build_tile_desc(input.shape, [1, tile_size], i, return_valid_size=True)
            out_start_indices, out_data_sizes = build_tile_desc(output.shape, [1, tile_size], i)
            input_desc = InputDependencyDesc(input, require_full=False, start_indices=in_start_indices,
                                             data_sizes=in_data_sizes)
            weight_desc = InputDependencyDesc(weight, require_full=True)
            out_desc = OutputTilingDesc(start_indices=out_start_indices, tile_sizes=out_data_sizes)
            inputs_dep = {input: input_desc, weight: weight_desc}
            outs_tile_mapping = {output: out_desc}
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params,
                                 inputs_dep, outs_tile_mapping))
        return tasks


@registry.register_task(op_type="qkv_pack_qk_norm_rope_split_v", task_cls=QKVPackQKNormRopeSplitVTask,
                        config_factory=qkv_pack_qk_norm_rope_split_v_config_factory,
                        codegen_func=codegen_qkv_pack_qk_norm_rope_split_v)
class QKVPackQKNormRopeSplitVTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        qkv, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = io_tensors[0]
        q_norm_rope, k_norm_rope, v = io_tensors[1]
        assert len(qkv.shape) == 4
        bs, seq_len, num_total_heads, head_dim = qkv.shape
        num_q_heads = q_norm_rope.shape[-2]
        num_kv_heads = k_norm_rope.shape[-2]
        group_size = num_q_heads // num_kv_heads
        BLOCK_HD = next_power_of_2(head_dim)
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config(BLOCK_HD=BLOCK_HD)
        kernel_config.BLOCK_SIZE_B = cdiv(bs * num_tiles_seq * num_total_heads, cls.NUM_SMS)
        num_tiles_seq = cdiv(seq_len, kernel_config.BLOCK_SEQ)
        B_blocks = cdiv(bs, kernel_config.BLOCK_SIZE_B)
        # num_tiles = B_blocks * num_tiles_seq * num_total_heads
        num_tiles = bs * num_tiles_seq * num_total_heads
        cls.log(f"QKVPackQKNormRopeSplitVTask Task: num_tiles = {num_tiles}")
        tasks = []
        BLOCK_SEQ = kernel_config.BLOCK_SEQ
        for tile_id in range(num_tiles):
            head_type = tile_id % (group_size + 2)
            tile_id_bs_seq = tile_id // num_total_heads
            bs_idx = tile_id_bs_seq // num_tiles_seq
            tild_id_seq = tile_id_bs_seq % num_tiles_seq
            head_group_id = (tile_id % num_total_heads) // (group_size + 2)
            if head_type == group_size + 1:  # v
                head_id_out = head_group_id
                head_id_input = head_id_out + num_q_heads + num_kv_heads
            elif head_type < group_size:  # q
                head_id_out = head_group_id * group_size + head_type
                head_id_input = head_id_out
            else:  # k
                head_id_out = head_group_id
                head_id_input = head_id_out + num_q_heads

            seq_tile_size = min(seq_len - tild_id_seq * BLOCK_SEQ, BLOCK_SEQ)
            input_desc = InputDependencyDesc(qkv, require_full=False,
                                             start_indices=(bs_idx, tild_id_seq * BLOCK_SEQ, head_id_input, 0),
                                             data_sizes=(1, seq_tile_size, 1, head_dim))
            out_desc = OutputTilingDesc(start_indices=(bs_idx, tild_id_seq * BLOCK_SEQ, head_id_out, 0),
                                        tile_sizes=(1, BLOCK_SEQ, 1, head_dim))
            empty_out_desc = OutputTilingDesc(start_indices=(0, 0, 0, 0), tile_sizes=(0, 0, 0, 0))
            if head_type < group_size:  # q
                outs_tile_mapping = {q_norm_rope: out_desc, k_norm_rope: empty_out_desc, v: empty_out_desc}
            elif head_type == group_size:
                outs_tile_mapping = {q_norm_rope: empty_out_desc, k_norm_rope: out_desc, v: empty_out_desc}
            else:
                outs_tile_mapping = {q_norm_rope: empty_out_desc, k_norm_rope: empty_out_desc, v: out_desc}
            inputs_dep = {qkv: input_desc}
            tasks.append(
                cls._create_task(layer_id, task_id, tile_id, num_tiles, kernel_config, dependency, io_tensors,
                                 extra_params, inputs_dep, outs_tile_mapping))
        return tasks