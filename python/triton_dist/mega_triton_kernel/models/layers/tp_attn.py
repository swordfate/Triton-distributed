import torch
from torch import nn
from ..paged_kv_cache import PagedKVCache
from ...tasks.utils import cdiv

try:
    import shmem as ash
except ImportError:
    ash = None
    print("No shmem!")

def shard_local(tensor: torch.Tensor, world_size: int, dim: int, local_rank: int):
    tensor_dim = tensor.shape[dim]
    tensor_slice = tensor_dim // world_size
    if tensor_dim % world_size != 0:
        raise ValueError(f"Tensor dimension {tensor_dim} is not divisible by world size {world_size}.")
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"Local rank {local_rank} is out of bounds for world size {world_size}.")
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"Dimension {dim} is out of bounds for tensor with {tensor.dim()} dimensions.")
    if tensor_slice == 0:
        raise ValueError(f"Tensor slice size is zero for tensor dimension {tensor_dim} and world size {world_size}.")
    return tensor.split(tensor_slice, dim=dim)[local_rank].contiguous()


# adapt from triton_dist/layers/nvidia/tp_attn.py
class TPAttnBuilder:
    """
    Tensor Parallel Attention.
    QKV Projection: Column Parallelism on weights (sharded over head dimension).
    Output Projection: Row Parallelism on weights.
    """

    def __init__(self, builder, layer_idx, head_dim=128, rank=0, world_size=8, group=None):
        self._builder = builder
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.head_dim = head_dim
        self.wqkv = None
        self.wo = None
        self.kv_cache = None
        self.layer_idx = layer_idx
        self.soft_cap = 0.0
        self.sm_scale = self.head_dim**-0.5

    def _init_parameters(self, self_attn: nn.Module, verbose=False):
        self.q_size = self_attn.q_proj.weight.shape[0] // self.world_size
        self.kv_size = self_attn.k_proj.weight.shape[0] // self.world_size
        wq = shard_local(self_attn.q_proj.weight.detach(), self.world_size, 0, self.rank)
        wk = shard_local(self_attn.k_proj.weight.detach(), self.world_size, 0, self.rank)
        wv = shard_local(self_attn.v_proj.weight.detach(), self.world_size, 0, self.rank)
        self.wqkv = torch.cat((wq, wk, wv), dim=0).to("cuda", non_blocking=True)  # [qkv_dim, hidden_size]
        self.wo = shard_local(self_attn.o_proj.weight.detach(), self.world_size, 1,
                              self.rank).to("cuda", non_blocking=True)
        self.q_head_num = self.q_size // self.head_dim
        # 分布式
        if self.world_size > 1:
            self.barrier_tensor = ash.aclshmem_create_tensor([self.world_size * 4 * 8], dtype=torch.int64, device_id=self.rank)
            self.barrier_intra_node = ash.aclshmem_create_tensor([self._builder.NUM_SMS * 2 * 8], dtype=torch.int64, device_id=self.rank)
            self.barrier_tensor.zero_()
            self.barrier_intra_node.zero_()

        self.ag_N_per_rank = self.wqkv.shape[0]
        self.K = self.wqkv.shape[1]
        self.dtype = self.wqkv.dtype

        if hasattr(self_attn, "q_norm"):
            self.q_norm_eps = self_attn.q_norm.variance_epsilon
            self.q_norm_w = self_attn.q_norm.weight.detach().to("cuda", non_blocking=True)
        if hasattr(self_attn, "k_norm"):
            self.k_norm_eps = self_attn.k_norm.variance_epsilon
            self.k_norm_w = self_attn.k_norm.weight.detach().to("cuda", non_blocking=True)
        assert hasattr(self_attn, "q_norm") and hasattr(self_attn, "k_norm")

        if verbose:
            print(f"[RANK {self.rank}] Attn initialized with parameters: qkv ({self.wqkv.shape}, o ({self.wo.shape}))")

        self.partial_out = None
        self.lse = None
        self.KV_SPLITS = None
        # self.qk_out = None
        # self.p_out = None
        # self.pv_out = None
    def build_fwd(self, x, cos_cache, sin_cache, kv_cache: PagedKVCache):
        """
        x: input tensor, shape [batch_size, q_len, hidden_size] (replicated on each rank)
        """
        key_cache, value_cache, block_tables, kv_lens = kv_cache.get_layer_kv_cache(self.layer_idx)
        self.max_length = kv_cache.max_length
        assert hasattr(self, 'q_norm_eps') and hasattr(self, 'k_norm_eps')
        assert len(x.shape) == 3 and x.dtype == torch.bfloat16
        batch_size, q_len, hidden_size = x.shape
        x = x.reshape(-1, hidden_size)
        num_tokens = batch_size * q_len

        qkv_proj_out = torch.zeros((num_tokens, self.wqkv.shape[0]), dtype=x.dtype, device=x.device)
        q_norm_rope = torch.zeros((batch_size, q_len, self.q_head_num, self.head_dim), dtype=x.dtype,
                                  device=x.device)
        if self.world_size > 1:
            # o_proj_out = self._builder.create_symm_tensor((num_tokens, hidden_size), x.dtype)
            # ar_out = torch.zeros(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
            # 分布式
            BLOCK_SIZE_B = 16 # TODO: same as BLOCK_SIZE_M of OProjTask
            prob_size_in_rank = (num_tokens + self.world_size - 1) // self.world_size
            b_loops = (prob_size_in_rank + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
            padded_B_in_rank = b_loops * BLOCK_SIZE_B
            peer_mem_elements = num_tokens * hidden_size + padded_B_in_rank * hidden_size
            peer_mem = ash.aclshmem_create_tensor([peer_mem_elements], dtype=x.dtype, device_id=self.rank)
            # o_proj_out = peer_mem[:num_tokens * hidden_size].view(num_tokens, hidden_size)
            o_proj_out = peer_mem.view(-1, hidden_size)
            ar_out = torch.zeros(num_tokens, hidden_size, dtype=x.dtype, device=x.device) # result of All-Reduce
        else:
            o_proj_out = torch.zeros((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
        
        # KV_SPLITS: 控制 flash decode split 的 KV 分片数
        # 目标1: 每片处理约 48 个 KV page (~3K tokens)，控制单 tile 计算量
        # 目标2: 总 tile 数 >= NUM_SMS * 3，保证 20 个 AICore 平均有 3+ 个 tile
        # 上限: 8，避免 partial_out/lse 过大和 combine 开销
        num_kv_heads = self.kv_size // self.head_dim
        num_q_heads_per_group = self.q_head_num // num_kv_heads
        BLOCK_H = 4
        head_groups = cdiv(self.q_head_num, min(num_q_heads_per_group, BLOCK_H))
        base_tiles = num_tokens * head_groups
        total_pages = (self.max_length + kv_cache.PAGE_SIZE - 1) // kv_cache.PAGE_SIZE
        target_pages_per_split = 48
        kv_splits_for_work = max(1, cdiv(total_pages, target_pages_per_split))
        target_tiles_per_sm = 3
        min_tiles = self._builder.NUM_SMS * target_tiles_per_sm
        kv_splits_for_dist = max(1, cdiv(min_tiles, base_tiles))
        self.KV_SPLITS = max(kv_splits_for_work, kv_splits_for_dist)
        self.KV_SPLITS = min(self.KV_SPLITS, 8)
        print(f"KV_SPLITS = {self.KV_SPLITS} (base_tiles={base_tiles})")
        self._builder.make_qkv_proj(x, self.wqkv, qkv_proj_out)
        # return qkv_proj_out.reshape(batch_size, q_len, hidden_size), qkv_proj_out
        qkv_proj_out_bsnh = qkv_proj_out.reshape(batch_size, q_len, -1, self.head_dim)
        self._builder.make_qk_norm_rope_update_kvcache(qkv_proj_out_bsnh, key_cache, value_cache, block_tables, kv_lens,
                                                       self.q_norm_w, self.k_norm_w, cos_cache, sin_cache, q_norm_rope,
                                                       self.q_norm_eps, self.k_norm_eps)
        attn_out = torch.zeros_like(q_norm_rope)

        # q_norm_rope = q_norm_rope.view(num_tokens, self.q_head_num, self.head_dim)
        # self.qk_out = torch.zeros((num_tokens, self.q_head_num, self.max_length), dtype=x.dtype, device=x.device)
        # self.p_out = torch.zeros((num_tokens, self.q_head_num, self.max_length), dtype=x.dtype, device=x.device)
        # self.pv_out = torch.zeros((num_tokens, self.q_head_num, self.head_dim), dtype=x.dtype, device=x.device)
        # self._builder.make_decode_gqa(q_norm_rope, key_cache, value_cache, block_tables, kv_lens, attn_out,
        #                               self.sm_scale, self.soft_cap)
                                    #   self.qk_out, self.p_out, self.pv_out, self.sm_scale, self.soft_cap)
        
        self.partial_out = torch.empty([batch_size, self.q_head_num, self.KV_SPLITS, self.head_dim], dtype=torch.float32,
                                  device=q_norm_rope.device)
        self.lse = torch.empty([batch_size, self.q_head_num, self.KV_SPLITS], dtype=torch.float32, device=q_norm_rope.device)
        # print(self.partial_out.shape)
        # print(self.lse.shape)
        self._builder.make_flash_decode(q_norm_rope, key_cache, value_cache, block_tables, kv_lens, attn_out,
                                        self.partial_out, self.lse, self.sm_scale, self.soft_cap, self.KV_SPLITS)
        attn_out_2d = attn_out.reshape(num_tokens, self.q_size)
        self._builder.make_o_proj(attn_out_2d, self.wo, o_proj_out)
        if self.world_size > 1:
            # self._builder.make_allreduce(o_proj_out, ar_out, double_input_buffer=True)
            # return ar_out.reshape(batch_size, q_len, hidden_size)
            # 分布式
            self._builder.make_allreduce_ascend(
                peer_mem, ar_out,
                barrier_global=self.barrier_tensor,
                barrier_intra=self.barrier_intra_node,
            )
            return ar_out.reshape(batch_size, q_len, hidden_size)

        return o_proj_out.reshape(batch_size, q_len, hidden_size)