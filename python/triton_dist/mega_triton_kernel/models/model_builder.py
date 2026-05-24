import torch
import importlib
import tempfile
import os
# import nvshmem
# import nvshmem.core

from ..core.code_generator import CodeGenerator, CodeGenOptions
from ..core.registry import registry
from ..core.task_base import TaskBase, DeviceProp, TaskDependency, TaskIDManager, MAX_NUM_TENSOR_DIMS
from ..core.builder import TaskBuilderBase
from ..core.graph import Graph
from typing import List, Dict, Any
# from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync
from ..core.scheduler import enque_tasks
from mega_triton_kernel_ascend.models.utils import logger
from mega_triton_kernel_ascend.tools.profiler import alloc_profiler_buffer, export_to_perfetto_trace, reset_profiler_buffer, parse_to_tracks
import triton

import acl, torch_npu

def is_multicast_ptr(tensor):
    """
    On unsupported platforms, `mc_ptr` return nullptr
    """
    return nvshmem.bindings.mc_ptr(nvshmem.core.Teams.TEAM_NODE, tensor.data_ptr()) != 0


def check_tensor_shape(tensor, shape):
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(shape, (list, tuple))
    tensor_shape = list(tensor.shape)
    assert len(tensor_shape) == len(shape), f"tensor shape mismatch, tensor shape = {tensor.shape}, shape = {shape}"
    for (x, y) in zip(tensor_shape, shape):
        assert x == y, f"tensor shape mismatch, tensor shape = {tensor.shape}, shape = {shape}"


def check_tensor_dim(tensor, ndim):
    assert isinstance(tensor, torch.Tensor)
    assert len(
        tensor.shape
    ) == ndim, f"tensor dim mismatch, tensor dim = {len(tensor.shape)}, ndim = {ndim}, shape = {tensor.shape}"


def check_tensor_dtype(tensor, dtype):
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == dtype, f"tensor dtype mismatch, expect {dtype}, but got {tensor.dtype}"


def check_contiguous(tensors):
    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors]
    for t in tensors:
        assert t.is_contiguous()


def check_alignment(tensors):
    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors]
    for t in tensors:
        assert t.data_ptr() % 16 == 0, f"data_ptr = {t.data_ptr()}"


class ModelBuilder:

    def __init__(self, rank=0, world_size=1, local_world_size=1, num_warps=4, enable_profiling=False,
                 enable_dep_opt=True, target_hw='gpu', enable_runtime_scheduler=False, enable_task_prefetch=False):
        self.reset()
        self._registry = registry
        self._code_generator = CodeGenerator()
        self._max_tensor_dim = MAX_NUM_TENSOR_DIMS
        # NUM_SMS = 4
        NUM_SMS = 20 if acl.get_soc_name()=='Ascend910B3' else 24
        self.NUM_SMS = NUM_SMS
        self.device_prop = DeviceProp(NUM_SMS=NUM_SMS)
        self.megakernel_tasks = []
        self.scoreboard = None
        self.wq_tensor = None  # work queue
        self.num_task_tensor = None  # num task in each work queue
        self.MAX_NUM_TILES_PER_OP = 1
        self.last_dependency = TaskDependency()
        self.max_layer_id = 0
        self.max_task_id = 0
        self.num_warps = num_warps
        self._metrics = {"memory": 0}
        self.world_size = world_size
        self.local_world_size = local_world_size
        self.rank = rank
        self.local_rank = self.rank % self.local_world_size
        assert self.world_size % self.local_world_size == 0
        assert self.world_size > 0 and self.local_world_size > 0
        self.all_symm_tensors = []
        # if self.world_size > 1:
        #     self.barrier_all_intra_node_buf = self.create_symm_tensor([
        #         world_size,
        #     ], torch.int32)
        #     self.barrier_all_intra_node_buf.zero_()
        #     torch.distributed.barrier()
        # else:
        #     self.barrier_all_intra_node_buf = None
        self.logger = logger
        self._enable_profiling = enable_profiling
        self._enable_dep_opt = enable_dep_opt
        self._enable_runtime_scheduler = enable_runtime_scheduler
        self._enable_task_prefetch = enable_task_prefetch
        self.target_hw = target_hw
        self.task_types_to_str = None
        self._graph = Graph()
        
        # 分布式
        self.model = None

    def create_symm_tensor(self, shape, dtype) -> torch.Tensor:
        tensor = nvshmem_create_tensor(shape, dtype)
        self.all_symm_tensors.append(tensor)
        return tensor

    def _update_metrics(self, op_type: str, io_tensors: List[List[torch.Tensor]], extra_params: Dict[str, Any] = {}):
        # avoid kv cache tensor being counted
        if "attn" in op_type or "kvcache" in op_type:
            return

        nbytes = 0
        all_tensors = io_tensors[0] + io_tensors[1]
        for ten in all_tensors:
            nbytes += ten.numel() * ten.element_size()

        if "memory" not in self._metrics:
            self._metrics["memory"] = 0
        self._metrics["memory"] += nbytes

    def get_memory_size(self):
        return self._metrics["memory"]

    def _update_tasks(self, tasks: List[TaskBase], do_not_update_dependency=False):
        print(f'len of tile tasks for current op = {len(tasks)}')
        assert len(tasks) > 0
        last_task = tasks[-1]
        if not do_not_update_dependency:
            self.last_dependency = TaskDependency(layer_id=last_task.layer_id, task_id=last_task.task_id, start_tiles=0,
                                                  end_tiles=last_task.num_tiles) # 是上一个op的全部tile
        self.megakernel_tasks += tasks
        for task in tasks:
            self.MAX_NUM_TILES_PER_OP = max(self.MAX_NUM_TILES_PER_OP, task.num_tiles)
            self.max_layer_id = max(task.layer_id, self.max_layer_id)
            self.max_task_id = max(task.task_id, self.max_task_id)

    def get_sm_activity(self):
        assert self._enable_profiling
        block_idx_to_tracks = parse_to_tracks(self.profile_buf)
        sb_wait_deps_task_type = None
        for k, v in self.task_types_to_str.items():
            if v == "scoreboard_wait_deps":
                sb_wait_deps_task_type = k
        assert sb_wait_deps_task_type is not None
        wait_deps_time = 0
        e2e_time = 0
        NUM_SMS = self.device_prop.NUM_SMS
        act_time = 0
        for i in range(NUM_SMS):
            assert i in block_idx_to_tracks.keys()
            tracks_list = block_idx_to_tracks[i]
            block_time = 0
            for track in tracks_list:
                if track.task_type == sb_wait_deps_task_type:
                    wait_deps_time += track.duration
                else:
                    act_time += track.duration
                block_time += track.duration
            e2e_time = max(e2e_time, block_time)
        self.logger.log(
            f"e2e_time = {e2e_time * 1.0 / 1e6} ms, avg_time = {act_time * 1.0 / NUM_SMS  / 1e6} ms, avg_wait_deps_per_block_time = {wait_deps_time * 1.0 / NUM_SMS  / 1e6} ms",
            level="debug")
        return act_time * 1.0 / (e2e_time * NUM_SMS)

    def get_task_builder(self, op_type: str) -> 'TaskBuilderBase':
        task_type = self._registry.get_op_mapping(op_type)
        if not task_type:
            raise ValueError(f"Unsupported op type: {op_type}")
        builder_cls = registry.get_builder(task_type)
        return builder_cls

    def _convert_op(self, op_type: str, layer_id: int, io_tensors: List[List[torch.Tensor]],
                    extra_params: Dict[str, Any] = {}) -> List[TaskBase]:
        assert len(io_tensors) == 2
        check_contiguous(io_tensors[0])
        check_contiguous(io_tensors[1])
        check_alignment(io_tensors[0])
        check_alignment(io_tensors[1])

        builder_cls = self.get_task_builder(op_type)
        # print(f'last_dependency = {self.last_dependency}')
        tasks = builder_cls.build_tasks(device_prop=self.device_prop, layer_id=layer_id,
                                        dependency=self.last_dependency, io_tensors=io_tensors,
                                        extra_params=extra_params, target_hw=self.target_hw)
        print(f'op_type = {op_type}')
        self._update_tasks(tasks)
        self._update_metrics(op_type, io_tensors, extra_params)
        self._graph.new_node(tasks=tasks, op_type=op_type, io_tensors=io_tensors, extra_params=extra_params)
        return tasks

    def _make_fc(self, op_type: str, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
                 layer_id: int = 0):
        check_tensor_dim(input, 2)
        check_tensor_dim(weight, 2)
        M, K = input.shape
        N, wK = weight.shape
        oM, oN = output.shape
        assert K == wK
        assert oM >= M and oN == N
        assert K % 32 == 0

        self._convert_op(op_type, layer_id, [[input, weight], [output]])

    def make_fc1(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("mlp_fc1", input, weight, output, layer_id)

    def make_fc2(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("mlp_fc2", input, weight, output, layer_id)

    def make_qkv_proj(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("qkv_proj", input, weight, output, layer_id)

    def make_o_proj(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("o_proj", input, weight, output, layer_id)

    def make_linear(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        check_tensor_dim(input, 2)
        check_tensor_dim(weight, 2)
        M, K = input.shape
        N, wK = weight.shape
        oM, oN = output.shape
        assert K == wK
        assert oM == M and oN == N
        self._convert_op("linear", layer_id, [[input, weight], [output]])

    def make_flash_decode(self, query, key_cache: torch.Tensor, value_cache: torch.Tensor, block_tables: torch.Tensor,
                          kv_lens: torch.Tensor, output: torch.Tensor, partial_out: torch.Tensor, lse: torch.Tensor, 
                          sm_scale=None, soft_cap=0.0, kv_split=2, layer_id: int = 0):
        """
            query: (batch, seq_len, num_q_heads, q_head_dim)
            key_cache: (MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, q_head_dim)
            value_cache: (MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, v_head_dim)
            block_tables: (batch, MAX_NUM_BLOCKS_PER_SEQ)
            kv_lens: (batch)
            output: (batch, seq_len, num_q_heads, v_head_dim)
        """
        check_tensor_dim(query, 4)
        check_tensor_dim(key_cache, 4)
        check_tensor_dim(value_cache, 4)
        check_tensor_dim(block_tables, 2)
        check_tensor_dim(kv_lens, 1)
        check_tensor_dim(output, 4)
        check_tensor_dtype(query, torch.bfloat16)
        check_tensor_dtype(key_cache, torch.bfloat16)
        check_tensor_dtype(value_cache, torch.bfloat16)
        check_tensor_dtype(block_tables, torch.int32)
        check_tensor_dtype(kv_lens, torch.int32)
        check_tensor_dtype(output, torch.bfloat16)
        assert query.shape[0] == kv_lens.shape[0]

        batch, q_seq_len, num_q_heads, q_head_dim = query.shape
        v_head_dim = value_cache.shape[-1]
        assert q_seq_len == 1, "currently flash decoding only support q_seq_len == 1"
        if sm_scale is None:
            sm_scale = q_head_dim**-0.5
        soft_cap = 0.0
        extra_params = {"sm_scale": sm_scale, "soft_cap": soft_cap, "NUM_KV_SPLITS": kv_split}
        # Assuming q_seq_len is 1, ignore the seq_len dimension to reduce the number of tensor dimensions (MAX_TENSOR_DIM = 4)
        self._convert_op("attn_split", layer_id,
                         [[query, key_cache, value_cache, block_tables, kv_lens], [partial_out, lse]], extra_params)
        self._convert_op("attn_combine", layer_id, [[kv_lens, partial_out, lse], [output]], extra_params)

    def make_decode_gqa_origin(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        kv_length: torch.Tensor,
        attn_out: torch.Tensor,
        qk_out: torch.Tensor,
        p_out: torch.Tensor,
        pv_out: torch.Tensor,
        sm_scale=None,
        soft_cap=0.0,
        layer_id: int = 0,
    ):

        check_tensor_dim(q, 3)
        check_tensor_dim(k_cache, 4)
        check_tensor_dim(v_cache, 4)
        check_tensor_dim(block_table, 2)
        check_tensor_dim(kv_length, 1)

        check_tensor_dim(attn_out, 3)

        check_tensor_dtype(q, torch.bfloat16)
        check_tensor_dtype(k_cache, torch.bfloat16)
        check_tensor_dtype(v_cache, torch.bfloat16)
        check_tensor_dtype(attn_out, torch.bfloat16)

        check_tensor_dtype(block_table, torch.int32)
        check_tensor_dtype(kv_length, torch.int32)

        token_num = q.shape[0]
        num_q_heads = q.shape[1]
        q_head_dim = q.shape[2]

        num_blocks = k_cache.shape[0]
        page_size = k_cache.shape[1]
        num_kv_heads = k_cache.shape[2]
        Lk = k_cache.shape[3]

        v_num_blocks = v_cache.shape[0]
        v_page_size = v_cache.shape[1]
        v_num_kv_heads = v_cache.shape[2]
        Lv = v_cache.shape[3]

        batch_size = kv_length.shape[0]
        max_num_blocks_per_seq = block_table.shape[1]

        assert q_head_dim == Lk, (
            f"q head dim ({q_head_dim}) must equal k head dim ({Lk})"
        )

        assert v_num_blocks == num_blocks, (
            f"v_cache num_blocks ({v_num_blocks}) must match k_cache num_blocks ({num_blocks})"
        )
        assert v_page_size == page_size, (
            f"v_cache page_size ({v_page_size}) must match k_cache page_size ({page_size})"
        )
        assert v_num_kv_heads == num_kv_heads, (
            f"v_cache num_kv_heads ({v_num_kv_heads}) must match k_cache num_kv_heads ({num_kv_heads})"
        )

        assert block_table.shape[0] == kv_length.shape[0]

        assert attn_out.shape[0] == token_num
        assert attn_out.shape[1] == num_q_heads
        assert attn_out.shape[2] == Lv

        assert num_q_heads % num_kv_heads == 0, (
            "num_q_heads must be divisible by num_kv_heads"
        )

        q_heads_per_kv_head = num_q_heads // num_kv_heads

        BLOCK_N = page_size
        BLOCK_H = 2
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0

        BLOCK_DV = triton.next_power_of_2(Lv)

        q_head_block_size = min(BLOCK_H, q_heads_per_kv_head)

        assert q_head_block_size >= 2, (
            f"q_head_block_size ({q_head_block_size}) must be >= 2. "
            "Current kernel uses parallel(0, 2), so q_heads_per_kv_head=1 is not supported."
        )
        assert q_head_block_size % 2 == 0, (
            f"q_head_block_size ({q_head_block_size}) must be even"
        )

        groups_per_kv_head = triton.cdiv(q_heads_per_kv_head, BLOCK_H)

        tile_cnt = (
            token_num
            * num_kv_heads
            * groups_per_kv_head
        )

        if sm_scale is None:
            sm_scale = Lk ** -0.5

        block_constants = {
            "sm_scale": sm_scale,

            "BLOCK_H": BLOCK_H,
            "BLOCK_DMODEL": BLOCK_DMODEL,
            "BLOCK_DPE": BLOCK_DPE,
            "BLOCK_DV": BLOCK_DV,
            "BLOCK_N": BLOCK_N,

            "Lk": Lk,
            "Lv": Lv,
            "MAX_NUM_BLOCKS_PER_SEQ": max_num_blocks_per_seq,

            "q_heads_per_kv_head": q_heads_per_kv_head,
            "q_head_num": num_q_heads,
            "kv_head_num": num_kv_heads,
            "q_head_block_size": q_head_block_size,

            # 如果调度器/TaskBuilder 需要 tile_cnt，可以保留；
            # 但当前 paged_gqa_fwd_kernel 签名里没有 tile_cnt，不要在 codegen 里传给 kernel。
            "tile_cnt": tile_cnt,
        }

        self._convert_op(
            "decode_gqa_origin",
            layer_id,
            [
                [
                    q,
                    k_cache,
                    v_cache,
                    block_table,
                    kv_length,
                ],
                [
                    attn_out,
                    qk_out,
                    p_out,
                    pv_out
                ],
            ],
            block_constants,
        )

    def make_decode_gqa(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        kv_length: torch.Tensor,
        attn_out: torch.Tensor,
        sm_scale=None,
        soft_cap=0.0,
        layer_id: int = 0,
    ):
        """
        Kernel io_tensors_ptr layout:

            [0] q
            [1] k_cache
            [2] v_cache
            [3] block_table
            [4] kv_length
            [5] Att_Out

        Tensor layout:

            q:        [batch, q_seq_len, q_head_num, Lk]
            k_cache:  [num_blocks, page_size, kv_head_num, Lk]
            v_cache:  [num_blocks, page_size, kv_head_num, Lv]
            Att_Out:  [batch, q_seq_len, q_head_num, Lv]
        """

        check_tensor_dim(q, 4)
        check_tensor_dim(k_cache, 4)
        check_tensor_dim(v_cache, 4)
        check_tensor_dim(block_table, 2)
        check_tensor_dim(kv_length, 1)

        check_tensor_dim(attn_out, 4)

        check_tensor_dtype(q, torch.bfloat16)
        check_tensor_dtype(k_cache, torch.bfloat16)
        check_tensor_dtype(v_cache, torch.bfloat16)
        check_tensor_dtype(attn_out, torch.bfloat16)

        check_tensor_dtype(block_table, torch.int32)
        check_tensor_dtype(kv_length, torch.int32)

        batch_size = q.shape[0]
        q_seq_len = q.shape[1]
        num_q_heads = q.shape[2]
        q_head_dim = q.shape[3]

        num_blocks = k_cache.shape[0]
        page_size = k_cache.shape[1]
        num_kv_heads = k_cache.shape[2]
        k_head_dim = k_cache.shape[3]

        v_num_blocks = v_cache.shape[0]
        v_page_size = v_cache.shape[1]
        v_num_kv_heads = v_cache.shape[2]
        Lv = v_cache.shape[3]

        Lk = k_head_dim

        max_num_blocks_per_seq = block_table.shape[1]

        assert q_head_dim == Lk, (
            f"q head dim ({q_head_dim}) must equal k head dim ({Lk})"
        )

        assert v_num_blocks == num_blocks, (
            f"v_cache num_blocks ({v_num_blocks}) must match k_cache num_blocks ({num_blocks})"
        )
        assert v_page_size == page_size, (
            f"v_cache page_size ({v_page_size}) must match k_cache page_size ({page_size})"
        )
        assert v_num_kv_heads == num_kv_heads, (
            f"v_cache num_kv_heads ({v_num_kv_heads}) must match k_cache num_kv_heads ({num_kv_heads})"
        )

        assert block_table.shape[0] == batch_size
        assert kv_length.shape[0] == batch_size

        assert attn_out.shape[0] == batch_size
        assert attn_out.shape[1] == q_seq_len
        assert attn_out.shape[2] == num_q_heads
        assert attn_out.shape[3] == Lv

        assert num_q_heads % num_kv_heads == 0, (
            "num_q_heads must be divisible by num_kv_heads"
        )

        q_heads_per_kv_head = num_q_heads // num_kv_heads

        BLOCK_N = page_size
        BLOCK_H = 2
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0

        BLOCK_DV = triton.next_power_of_2(Lv)

        q_head_block_size = min(BLOCK_H, q_heads_per_kv_head)

        assert q_head_block_size >= 2, (
            f"q_head_block_size ({q_head_block_size}) must be >= 2. "
            "Current kernel uses parallel(0, 2), so q_heads_per_kv_head=1 is not supported."
        )
        assert q_head_block_size % 2 == 0, (
            f"q_head_block_size ({q_head_block_size}) must be even"
        )

        groups_per_kv_head = triton.cdiv(q_heads_per_kv_head, BLOCK_H)

        tile_cnt = (
            batch_size
            * q_seq_len
            * num_kv_heads
            * groups_per_kv_head
        )

        if sm_scale is None:
            sm_scale = Lk ** -0.5

        block_constants = {
            "sm_scale": sm_scale,

            "BLOCK_H": BLOCK_H,
            "BLOCK_DMODEL": BLOCK_DMODEL,
            "BLOCK_DPE": BLOCK_DPE,
            "BLOCK_DV": BLOCK_DV,
            "BLOCK_N": BLOCK_N,

            "Lk": Lk,
            "Lv": Lv,
            "Q_SEQ_LEN": q_seq_len,
            "MAX_NUM_BLOCKS_PER_SEQ": max_num_blocks_per_seq,

            "q_heads_per_kv_head": q_heads_per_kv_head,
            "q_head_num": num_q_heads,
            "kv_head_num": num_kv_heads,
            "q_head_block_size": q_head_block_size,

            # 如果调度器/TaskBuilder 需要 tile_cnt，可以保留；
            # 但当前 paged_gqa_fwd_kernel 签名里没有 tile_cnt，不要在 codegen 里传给 kernel。
            # "tile_cnt": tile_cnt,

            # "stride_block_table_batch": block_table.stride(0),

            # # q layout: [B, S, QH, Lk]
            # "stride_qbs": q.stride(0),
            # "stride_qs": q.stride(1),
            # "stride_qh": q.stride(2),

            # # k_cache layout: [num_blocks, page_size, KVH, Lk]
            # "stride_buf_kbs": k_cache.stride(0),
            # "stride_buf_kpage": k_cache.stride(1),
            # "stride_buf_kh": k_cache.stride(2),

            # # v_cache layout: [num_blocks, page_size, KVH, Lv]
            # "stride_buf_vbs": v_cache.stride(0),
            # "stride_buf_vpage": v_cache.stride(1),
            # "stride_buf_vh": v_cache.stride(2),

            # # attn_out layout: [B, S, QH, Lv]
            # "stride_mid_ob": attn_out.stride(0),
            # "stride_mid_os": attn_out.stride(1),
            # "stride_mid_oh": attn_out.stride(2),
        }

        self._convert_op(
            "decode_gqa",
            layer_id,
            [
                [
                    q,
                    k_cache,
                    v_cache,
                    block_table,
                    kv_length,
                ],
                [
                    attn_out,
                ],
            ],
            block_constants,
        )


    def make_silu_mul_up(self, fc1_out, act_out, layer_id=0):
        check_tensor_dim(fc1_out, 2)
        check_tensor_dim(act_out, 2)
        M, N = fc1_out.shape
        assert act_out.shape[0] == M
        assert act_out.shape[1] * 2 == N
        self._convert_op("silu_mul_up", layer_id, [[fc1_out], [act_out]])

    def make_qk_norm_rope_update_kvcache(self, qkv: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                                         block_tables: torch.Tensor, kv_lens: torch.Tensor, q_rms_weight: torch.Tensor,
                                         k_rms_weight: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor,
                                         q_norm_rope: torch.Tensor, q_rms_eps: float = 1e-6, k_rms_eps: float = 1e-6,
                                         rope_theta: int = 1000000, layer_id=0):
        """
            this op assume that kv_lens has been update (kv_lens = history_kv_len + seq_len(qkv.shape[1]))
            inplace update new kv to key_cache/value_cache
            cos_cache/sin_cache: [batch, seq_len, head_dim]
            rms_weight: [head_dim]
        """
        check_tensor_dim(qkv, 4)
        check_tensor_dim(key_cache, 4)
        check_tensor_dim(value_cache, 4)
        check_tensor_dim(block_tables, 2)
        check_tensor_dim(kv_lens, 1)
        check_tensor_dim(q_norm_rope, 4)
        check_tensor_dim(cos_cache, 3)
        check_tensor_dim(sin_cache, 3)
        check_tensor_dim(q_rms_weight, 1)
        check_tensor_dim(k_rms_weight, 1)
        check_tensor_dtype(qkv, torch.bfloat16)
        check_tensor_dtype(key_cache, torch.bfloat16)
        check_tensor_dtype(value_cache, torch.bfloat16)
        check_tensor_dtype(block_tables, torch.int32)
        check_tensor_dtype(kv_lens, torch.int32)
        check_tensor_dtype(q_norm_rope, torch.bfloat16)
        check_tensor_dtype(cos_cache, torch.float32)
        check_tensor_dtype(sin_cache, torch.float32)
        check_tensor_dtype(q_rms_weight, torch.bfloat16)
        check_tensor_dtype(k_rms_weight, torch.bfloat16)

        assert qkv.shape[0] == kv_lens.shape[0]
        assert cos_cache.shape[0] == sin_cache.shape[0]
        assert cos_cache.shape[0] == qkv.shape[0] or cos_cache.shape[0] == 1
        extra_params = {"q_rms_eps": q_rms_eps, "k_rms_eps": k_rms_eps, "rope_theta": rope_theta}
        self._convert_op("qk_norm_rope_update_kvcache", layer_id,
                         [[qkv, block_tables, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache],
                          [key_cache, value_cache, q_norm_rope]], extra_params)

    def make_qkv_pack_qk_norm_rope_split_v(self, qkv: torch.Tensor, kv_lens: torch.Tensor, q_rms_weight: torch.Tensor,
                                           k_rms_weight: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor,
                                           q_norm_rope: torch.Tensor, k_norm_rope: torch.Tensor, v: torch.Tensor,
                                           q_rms_eps: float, k_rms_eps: float, layer_id=0):
        """
            Inputs:
                qkv: [bs, seq_len, num_q_heads + 2 * num_kv_heads, head_dim]
                kv_lens: [bs]
                q/k_rms_weight: [head_dim]
                cos_cache/sin_cache: [MAX_SEQ_LEN, head_dim] or [1, MAX_SEQ_LEN, head_dim]

            Outputs:
                q_norm_rope: [bs, seq_len, num_q_heads, head_dim]
                k_norm_rope: [bs, seq_len, num_kv_heads, head_dim]
                v: [bs, seq_len, num_kv_heads, head_dim]
            
            Formula:
                q, k, v = qkv.split([num_q_heads, num_kv_heads, num_kv_heads], dim=-2)
                q_norm_rope = rope(rms_norm(q, q_rms_weight, q_rms_eps), cos_cache, sin_cache)
                k_norm_rope = rope(rms_norm(k, k_rms_weight, k_rms_eps), cos_cache, sin_cache)
        """
        check_tensor_dim(qkv, 4)
        check_tensor_dim(q_norm_rope, 4)
        check_tensor_dim(k_norm_rope, 4)
        bs, seq_len, num_total_heads, head_dim = qkv.shape
        num_q_heads = q_norm_rope.shape[-2]
        num_kv_heads = k_norm_rope.shape[-2]
        assert num_total_heads == num_q_heads + 2 * num_kv_heads
        check_tensor_shape(q_norm_rope, (bs, seq_len, num_q_heads, head_dim))
        check_tensor_shape(k_norm_rope, (bs, seq_len, num_kv_heads, head_dim))
        check_tensor_shape(v, (bs, seq_len, num_kv_heads, head_dim))
        check_tensor_shape(kv_lens, (bs, ))
        assert len(sin_cache.shape) == len(cos_cache.shape)
        assert sin_cache.shape == cos_cache.shape
        assert sin_cache.shape[-1] == head_dim

        check_tensor_dtype(q_norm_rope, torch.bfloat16)
        check_tensor_dtype(k_norm_rope, torch.bfloat16)
        check_tensor_dtype(cos_cache, torch.float32)
        check_tensor_dtype(sin_cache, torch.float32)
        check_tensor_dtype(q_rms_weight, torch.bfloat16)
        check_tensor_dtype(k_rms_weight, torch.bfloat16)

        extra_params = {"q_rms_eps": q_rms_eps, "k_rms_eps": k_rms_eps}
        self._convert_op(
            "qkv_pack_qk_norm_rope_split_v", layer_id,
            [[qkv, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache], [q_norm_rope, k_norm_rope, v]],
            extra_params)

    def make_rms_norm(self, input: torch.Tensor, rms_weight: torch.Tensor, output: torch.Tensor, rms_eps: float = 1e-6,
                      layer_id=0):
        check_tensor_dtype(input, torch.bfloat16)
        check_tensor_dtype(rms_weight, torch.bfloat16)
        check_tensor_dtype(output, torch.bfloat16)
        check_tensor_dim(rms_weight, 1)
        # reshape to 2d tensor
        input = input.reshape(-1, input.shape[-1])
        output = output.reshape(-1, input.shape[-1])

        assert input.shape == output.shape
        assert input.shape[-1] == rms_weight.shape[0]
        extra_params = {"rms_eps": rms_eps}
        self._convert_op("rms_norm", layer_id, [[input, rms_weight], [output]], extra_params)

    def make_add(self, lhs: torch.Tensor, rhs: torch.Tensor, output: torch.Tensor, layer_id=0):
        check_tensor_dtype(lhs, torch.bfloat16)
        check_tensor_dtype(rhs, torch.bfloat16)
        check_tensor_dtype(output, torch.bfloat16)

        assert lhs.shape == rhs.shape and lhs.shape == output.shape
        lhs = lhs.reshape(-1)
        rhs = rhs.reshape(-1)
        output = output.reshape(-1)

        self._convert_op("add", layer_id, [[lhs, rhs], [output]])

    def make_barrier_all_intra_node(self, wait_inputs=None, layer_id=0):
        """
            `wait_inputs` is used to build data dependency.
        """
        assert self.world_size > 1
        wait_inputs = [] if wait_inputs is None else wait_inputs
        extra_params = {"local_rank": self.local_rank, "local_world_size": self.local_world_size}
        self._convert_op("barrier_all_intra_node", layer_id,
                         [[self.barrier_all_intra_node_buf] + wait_inputs, wait_inputs], extra_params)

    def make_allreduce(self, input: torch.Tensor, output: torch.Tensor, double_input_buffer=False, layer_id=0):
        """
            if double_input_buffer is True, user needs to ensure that the input of two consecutive allreduce are completely different buffers,
            otherwise, the output may be wrong.
        """
        assert self.world_size > 1
        if not is_multicast_ptr(input):
            raise ValueError(
                "The input tensor needs to be a symmetric buffer and AR is only supported in Hopper and later GPU architectures"
            )
        assert os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "1") == "0"  # for multicast
        input = input.reshape(-1)
        output = output.reshape(-1)
        nbytes = input.numel() * input.element_size()
        assert nbytes % 128 == 0
        # assert input.shape == output.shape
        assert input.dtype == output.dtype
        self.make_barrier_all_intra_node(wait_inputs=[input], layer_id=layer_id)
        self._convert_op("allreduce", layer_id, [[input], [output]])
        if not double_input_buffer:
            self.make_barrier_all_intra_node(wait_inputs=[output], layer_id=layer_id)
    
    def make_allreduce_ascend(self, input, output, barrier_global, barrier_intra, layer_id=0):
        assert self.world_size > 1
        input = input.reshape(-1)
        # output = output.reshape(-1)
        nbytes = input.numel() * input.element_size()
        # assert nbytes % 128 == 0
        # assert input.shape == output.shape
        assert input.dtype == output.dtype
        
        extra_params = {
            "rank": self.rank,
            "world_size": self.world_size,
            # "barrier_global": barrier_global,
            # "barrier_intra": barrier_intra,
        }
        self._convert_op("allreduce_ascend", layer_id, 
                        io_tensors=[[input, barrier_global, barrier_intra], [output]], 
                        extra_params=extra_params)

    def make_gather_all(self, input, output, barrier_global, layer_id=0):
        assert self.world_size > 1
        input = input.reshape(-1)
        # output = output.reshape(-1)
        nbytes = input.numel() * input.element_size()
        # assert nbytes % 128 == 0
        assert input.dtype == output.dtype
        
        extra_params = {
            "rank": self.rank,
            "world_size": self.world_size,
        }
        self._convert_op("gather_all", layer_id, 
                        io_tensors=[[input, barrier_global], [output]], 
                        extra_params=extra_params)

    def make_prefetch(self, weight: torch.tensor, layer_id=0):
        io_tensors = [[weight], []]
        check_contiguous(io_tensors[0])
        check_contiguous(io_tensors[1])
        check_alignment(io_tensors[0])
        check_alignment(io_tensors[1])
        assert weight.dtype == torch.bfloat16
        check_tensor_dim(weight, 2)
        check_tensor_dtype(weight, torch.bfloat16)
        assert weight.shape[1] % 32 == 0

        builder_cls = self.get_task_builder("prefetch")
        tasks = builder_cls.build_tasks(device_prop=self.device_prop, layer_id=layer_id,
                                        dependency=TaskDependency(),  # no dependency
                                        io_tensors=io_tensors, extra_params={})
        self._update_tasks(tasks, do_not_update_dependency=True)

    def reset(self):
        TaskIDManager.reset_all_ids()
    
    def compile(self):
        self.logger.log(f"num_total_tasks = {len(self.megakernel_tasks)}", level="debug")
        print(f'total dep tasks num before _graph.to_tasks() = {sum([len(t.dependency) for t in self.megakernel_tasks])}')
        if self._enable_dep_opt:
            megakernel_tasks = self._graph.to_tasks()
        else:
            megakernel_tasks = self.megakernel_tasks
        print(f'total dep tasks num after _graph.to_tasks() = {sum([len(t.dependency) for t in self.megakernel_tasks])}')
        # 动态调度: num_sms=1 将所有 task 放入一个全局平面队列
        if self._enable_runtime_scheduler:
            num_sms = 1
        else:
            num_sms = self.device_prop.NUM_SMS
        self.wq_tensor, self.num_tasks_tensor, self.scoreboard, self.task_deps_tensor = enque_tasks(
            num_sms, megakernel_tasks, "round_robin",
            enable_dependency_opt=not self._enable_runtime_scheduler)
        # scbd中的每行cacheline对齐
        self.MAX_NUM_TILES_PER_OP = (self.MAX_NUM_TILES_PER_OP + 127) // 128 * 128
        self.scoreboard = torch.zeros((self.max_layer_id + 1, self.max_task_id + 1, self.MAX_NUM_TILES_PER_OP),
                                      dtype=torch.int32, device=torch.cuda.current_device())
        print(f'task_deps_tensor.shape = {self.task_deps_tensor.shape}')
        print(f'scoreboard.shape = {self.scoreboard.shape}')
        codegen_options = CodeGenOptions(
            enable_profiling=self._enable_profiling,
            enable_runtime_scheduler=self._enable_runtime_scheduler,
            enable_task_prefetch=self._enable_task_prefetch,
            target_hw=self.target_hw,
        )
        src, task_types_to_str = self._code_generator.generate_code(self.megakernel_tasks, codegen_options)
        # task_types_to_str = {0: 'RMSNormTask', 1: 'QKVProjTask', 2: 'QKNormRopeUpdateKVCacheTask', 3: 'AttnSplitTask', 
        # 4: 'AttnCombineTask', 5: 'OProjTask', 9: 'AddTask', 6: 'MLPFC1Task', 7: 
        # 'SiLUMulUpTask', 8: 'MLPFC2Task', 10: 'LinearTask', 11: 'scoreboard_wait_deps', 12: 'task_decoding'}
        print(src)
        self.logger.log(src, level="info")
        print(f'task_types_to_str = {task_types_to_str}')
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
            tmp.write(src.encode('utf-8'))
            tmp_path = tmp.name
        
        module_name = os.path.basename(tmp_path)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._gen_kernel = module.MEGA_TRITON_KERNEL
        self.task_types_to_str = task_types_to_str
        max_num_profile_slots = (self.wq_tensor.shape[0] + 4) * self.wq_tensor.shape[1] * 64
        self.logger.log(f"max_num_profile_slots = {max_num_profile_slots}", level="debug")
        if self._enable_profiling:
            self.profile_buf = alloc_profiler_buffer(max_num_profile_slots)
        else:
            self.profile_buf = None

    def dump_trace(self, trace_file_prefix="MEGA_KERNEL_TRACE"):
        if self._enable_profiling:
            profiler_dir = os.environ.get("MEGA_KERNEL_PRODILER_DIR", "./prof")
            os.makedirs(profiler_dir, exist_ok=True)
            trace_file = os.path.join(profiler_dir, f"{trace_file_prefix}_RANK_{self.rank}")
            export_to_perfetto_trace(self.profile_buf, self.task_types_to_str, trace_file)
        else:
            self.logger.log("profiler not enabled, please set enable_profiling=True", level="warning")

    def run(self):
        grid = lambda META: (self.device_prop.NUM_SMS, ) # NUM_SMS个线程块
        debug_counts = torch.zeros((self.max_layer_id + 1, self.max_task_id + 1, self.MAX_NUM_TILES_PER_OP),
                                      dtype=torch.int32, device=torch.cuda.current_device())
        # 动态调度: 全局原子计数器, 每个 SM 通过 atomic_add 抢下一个 task
        self._work_queue_start = torch.empty((1,), dtype=torch.int32, device=torch.cuda.current_device())
        if self._enable_runtime_scheduler:
            self._work_queue_start.fill_(0)
        if self._enable_profiling:
            assert self.profile_buf is not None
            reset_profiler_buffer(self.profile_buf)
            kernel_args = [
                self.profile_buf,
            ]
            if self._enable_runtime_scheduler:
                kernel_args.append(self._work_queue_start)
            kernel_args += [
                self.wq_tensor,
                self.num_tasks_tensor,
                self.scoreboard,
                self.task_deps_tensor,
            ]
            self._gen_kernel[grid](
                *kernel_args,
                INT_PER_DEPS=self.task_deps_tensor.shape[1],
                INT_PER_TASK=self.wq_tensor.shape[2],
                MAX_TASK_ID=self.scoreboard.shape[1],
                MAX_NUM_TILES_PER_OP=self.scoreboard.shape[2],
                MAX_NUM_TENSOR_DIMS=self._max_tensor_dim,
                NUM_SMS=self.device_prop.NUM_SMS,
                num_warps=self.num_warps,
                debug_counts=debug_counts,
            )
        else:
            kernel_args = []
            if self._enable_runtime_scheduler:
                kernel_args.append(self._work_queue_start)
            kernel_args += [
                self.wq_tensor,
                self.num_tasks_tensor,
                self.scoreboard,
                self.task_deps_tensor,
            ]
            self._gen_kernel[grid](
                *kernel_args,
                INT_PER_DEPS=self.task_deps_tensor.shape[1],
                INT_PER_TASK=self.wq_tensor.shape[2],
                MAX_TASK_ID=self.scoreboard.shape[1],
                MAX_NUM_TILES_PER_OP=self.scoreboard.shape[2],
                MAX_NUM_TENSOR_DIMS=self._max_tensor_dim,
                NUM_SMS=self.device_prop.NUM_SMS,
                num_warps=self.num_warps,
                debug_counts=debug_counts,
            )
        # print(f'scoreboard end run magekernel: {self.scoreboard}')
        # print(f'debug_counts end run magekernel: {debug_counts}')
        # is_valid = self.check_dependencies_nonzero(debug_counts, self.task_deps_tensor)
        # print(f"Check result: {is_valid}")

        self.scoreboard.zero_()
        
        # 分布式
        if self.world_size > 1:
            for idx in range(self.model.num_layers):
                self.model.layers[idx].attn.barrier_tensor.zero_()
                self.model.layers[idx].attn.barrier_intra_node.zero_()
                self.model.layers[idx].mlp.barrier_tensor.zero_()
                self.model.layers[idx].mlp.barrier_intra_node.zero_()
                # self.model.layers[idx].attn.qk_out.zero_()
                # self.model.layers[idx].attn.p_out.zero_()
                # self.model.layers[idx].attn.pv_out.zero_()
            self.model.barrier_tensor.zero_()
        torch_npu.npu.synchronize()

    def finalize(self):
        for ten in self.all_symm_tensors:
            nvshmem_free_tensor_sync(ten)

    def check_dependencies_nonzero(self, status_tensor, task_deps_tensor):
        """
        检查 status_tensor 在 task_deps_tensor 指定的位置上，数值是否都非 0
        """
        flat_status = status_tensor.view(-1)
        
        all_indices_list = []
        
        # 遍历每一行依赖描述
        for row in task_deps_tensor:
            start = row[0].item()
            end = row[1].item() # 后面的那个下标不取到
            
            if end > start:
                # 生成 start 到 end-1 的索引
                rng = torch.arange(start, end, device=status_tensor.device, dtype=torch.long)
                all_indices_list.append(rng)
                
        if not all_indices_list:
            print("没有需要检查的依赖区间。")
            return True

        # 3. 拼接成一个大的索引向量
        all_indices = torch.cat(all_indices_list)
        
        # 4. 查值并判断
        if all_indices.max() >= flat_status.numel():
            print(f"[Error] Index out of bounds! Max index {all_indices.max()} > Size {flat_status.numel()}")
            return False

        selected_values = flat_status[all_indices]
        
        # 判断是否全是非零
        all_nonzero = (selected_values != 0).all().item()
        
        if not all_nonzero:
            # 统计有多少个 0
            num_zeros = (selected_values == 0).sum().item()
            print(f"[Check Failed] Found {num_zeros} zero values in the specified ranges.")
            return False
            
        return True