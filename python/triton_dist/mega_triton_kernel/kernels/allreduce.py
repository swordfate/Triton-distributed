import triton
import triton.language as tl
# from triton.language.extra.cuda.language_extra import tid, __syncthreads
# from .task_context import TaskBaseInfo, Scoreboard
# from triton.language.extra.cuda.language_extra import (st_v4_b32, multimem_ld_reduce_v4)
# from triton.language.extra.cuda.utils import num_warps
from torch_npu.contrib import transfer_to_npu
from .task_context_utils import *

import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cann.extension import sub_vec_id

@triton.jit
def gemm_swizzle2d_Nz(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
):
    '''gemm swizzle Nz'''
    data_row_loop_num = tl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = tl.cdiv(data_col_shape, tile_col_shape)
    col_loop_num = tl.cdiv(data_col_loop_num, swizzle_offset)
    n_tile_idx = iter_id // (swizzle_offset * data_row_loop_num)
    m_n_tile_idx = iter_id % (swizzle_offset * data_row_loop_num)
    n_tile_size = swizzle_offset
    if n_tile_idx == col_loop_num - 1:
        n_tile_size = data_col_loop_num - swizzle_offset * n_tile_idx
    data_row_idx = m_n_tile_idx // n_tile_size
    data_col_idx = n_tile_idx * swizzle_offset + m_n_tile_idx % n_tile_size
    if n_tile_idx % 2 == 1:
        data_row_idx = data_row_loop_num - data_row_idx - 1
    return data_row_idx, data_col_idx
@triton.jit
def allreduce_task_compute(
    tile_id_or_start, 
    io_tensors_ptr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    scoreboard_ptr,
    layer_id,
    task_id,
    TILE_READY_SIGNAL: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr
):
    
    peer_mem_tensor      = task_base_info_get_tensor(io_tensors_ptr, 0, MAX_NUM_TENSOR_DIMS)
    barrier_intra_tensor = task_base_info_get_tensor(io_tensors_ptr, 1, MAX_NUM_TENSOR_DIMS)
    barrier_tensor       = task_base_info_get_tensor(io_tensors_ptr, 2, MAX_NUM_TENSOR_DIMS)
    c_tensor             = task_base_info_get_tensor(io_tensors_ptr, 3, MAX_NUM_TENSOR_DIMS)
    extra_params         = task_base_info_get_extra_params_ptr(io_tensors_ptr, 4, MAX_NUM_TENSOR_DIMS)

    B_dim = tensor_desc_size(c_tensor, 0)
    H_dim = tensor_desc_size(c_tensor, 1, 16)
    
    peer_mem_ptr = tensor_desc_data_ptr(peer_mem_tensor, tl.bfloat16)
    barrier_intra_ptr = tensor_desc_data_ptr(barrier_intra_tensor, tl.int64)
    barrier_ptr  = tensor_desc_data_ptr(barrier_tensor, tl.int64)
    c_ptr        = tensor_desc_data_ptr(c_tensor, tl.bfloat16)
    
    rank = tl.load(extra_params).to(tl.int32)
    rank_size = tl.load(extra_params + 1).to(tl.int32)
    stride_cb = H_dim
    stride_ch = 1

    # allreduce_tile_compute(
    #     tile_id=tile_id_or_start,
    #     c_ptr=c_ptr, peer_mem_ptr=peer_mem_ptr, rank=rank, rank_size=rank_size,
    #     barrier_intra_node_ptr=barrier_intra_ptr, barrier_ptr=barrier_ptr,
    #     B_dim=B_dim, H_dim=H_dim,
    #     stride_cb=stride_cb, stride_ch=stride_ch,
    #     BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H
    # )
    tile_id = tile_id_or_start
    barrier_intra_node_ptr = barrier_intra_ptr
    
    
    aivIndex = sub_vec_id() 
    ncore = tl.num_programs(axis=0) # 总的 AICore 数量

    problemSize_in_rank_b = (B_dim + rank_size - 1) // rank_size
    bLoops = tl.cdiv(problemSize_in_rank_b, BLOCK_SIZE_B)
    hLoops = tl.cdiv(H_dim, BLOCK_SIZE_H)

    # 所有的通信与显存读写操作放置在 aivIndex == 0 中执行
    if aivIndex == 0:
        # =========================================================================
        # PRE-PHASE: Notify Peers Input is Ready
        # =========================================================================
        barrier_addr = barrier_ptr + (0 * rank_size + rank) * 8
        for r in range(tile_id, rank_size, ncore):
            dl.notify(barrier_addr, r, signal=1, sig_op="set", comm_scope="intra_node")

        rs_buffer_ptr = peer_mem_ptr + B_dim * H_dim
        
        # =========================================================================
        # PHASE 1: Reduce-Scatter (寄存器内循环累加)
        # =========================================================================
        total_rs_blocks_spatial = bLoops * hLoops

        barrier_addr = barrier_ptr + (0 * rank_size) * 8
        rs_token = dl.wait(barrier_addr, rank_size, scope="gpu", _semantic="acquire", waitValue=1)

        for idx in range(tile_id, total_rs_blocks_spatial, ncore):
            b_id_in_comm, h_id_in_comm = gemm_swizzle2d_Nz(
                idx, bLoops * BLOCK_SIZE_B, H_dim, BLOCK_SIZE_B, BLOCK_SIZE_H
            )
            
            offs_cb = (
                rank * problemSize_in_rank_b 
                + b_id_in_comm * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
            )
            offs_ch = h_id_in_comm * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
            
            mask_b = (offs_cb < (rank + 1) * problemSize_in_rank_b) & (offs_cb < B_dim)
            mask_h = offs_ch < H_dim
            mask = mask_b[:, None] & mask_h[None, :]

            # 在高速寄存器中累加
            acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32)
            for target_rank in range(rank_size):
                remote_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                remote_ptr = dl.consume_token(remote_ptr, rs_token) 
                remote_ptrs = remote_ptr + offs_cb[:, None] * H_dim + offs_ch[None, :]
                c_temp = tl.load(remote_ptrs, mask=mask, other=0.0)
                acc += c_temp
            
            acc = acc.to(tl.bfloat16)
            # 写入 rs_buffer
            offs_cb_relative = b_id_in_comm * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
            rs_offs = offs_cb_relative[:, None] * H_dim + offs_ch[None, :]
            local_rs_ptrs = rs_buffer_ptr + rs_offs
            acc = acc.to(tl.bfloat16)
            tl.store(local_rs_ptrs, acc, mask=mask)
            
            # 写入 C 矩阵
            c_offs = stride_cb * offs_cb[:, None] + stride_ch * offs_ch[None, :]
            c_dst_ptrs = c_ptr + c_offs
            tl.store(c_dst_ptrs, acc, mask=mask)

        barrier_intra_node_addr = barrier_intra_node_ptr + tile_id * 8
        dl.notify(barrier_intra_node_addr, rank, signal=1, sig_op="set", comm_scope="intra_node")
        intrasync_token = dl.wait(barrier_intra_node_ptr, ncore, scope="gpu", _semantic="acquire", waitValue=1)
        
        # 当前 rank 内部计算完成，通知目标 rank
        for r in range(tile_id, rank_size, ncore):
            barrier_addr = barrier_ptr + (1 * rank_size + rank) * 8
            dl.notify(barrier_addr, r, signal=1, sig_op="set", comm_scope="intra_node")

        total_rs_blocks = bLoops * hLoops * rank_size
        # 等待所有 rank 的 Phase 1 结束
        wait_addr = barrier_ptr + (1 * rank_size) * 8
        p1_done_token = dl.wait(wait_addr, rank_size, scope="gpu", _semantic="acquire", waitValue=1)

        # =========================================================================
        # PHASE 2: All-Gather (直接拉取远程 RS Buffers)
        # =========================================================================
        for block_id in range(tile_id, total_rs_blocks, ncore):
            swizzle_b, swizzle_h = gemm_swizzle2d_Nz(
                block_id, rank_size * BLOCK_SIZE_B * bLoops, H_dim, BLOCK_SIZE_B, BLOCK_SIZE_H
            )
            src_rank_idx = swizzle_b // bLoops
            b_id_in_comm = swizzle_b % bLoops 
            
            if src_rank_idx != rank:
                remote_peer_ptr = dl.symm_at(peer_mem_ptr, src_rank_idx)
                ready_base_ptr = dl.consume_token(remote_peer_ptr, intrasync_token)
                ready_base_ptr = dl.consume_token(ready_base_ptr, p1_done_token)
                
                rs_buffer_offset = B_dim * H_dim
                offs_b_relative = b_id_in_comm * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
                offs_h = swizzle_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
                rs_offs = offs_b_relative[:, None] * H_dim + offs_h[None, :]
                
                remote_rs_ptrs = ready_base_ptr + rs_buffer_offset + rs_offs
                
                offs_b_absolute = (src_rank_idx * problemSize_in_rank_b 
                                + b_id_in_comm * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B))
                mask_b = (offs_b_absolute < (src_rank_idx + 1) * problemSize_in_rank_b) & (offs_b_absolute < B_dim)
                mask_h = offs_h < H_dim
                mask = mask_b[:, None] & mask_h[None, :]
                
                gathered_val = tl.load(remote_rs_ptrs, mask=mask, other=0.0)
                
                c_offs = stride_cb * offs_b_absolute[:, None] + stride_ch * offs_h[None, :]
                c_dst_ptrs = c_ptr + c_offs
                tl.store(c_dst_ptrs, gathered_val, mask=mask)

        scoreboard_release_tile_flat(
            scoreboard_ptr, 
            layer_id=layer_id, 
            task_id=task_id,
            tile_id=tile_id_or_start,
            TILE_READY_SIGNAL=TILE_READY_SIGNAL,
            MAX_TASK_ID=MAX_TASK_ID,
            MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
        )

@triton.jit
def gather_all_compute(
    tile_id_or_start, 
    io_tensors_ptr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    scoreboard_ptr,
    layer_id,
    task_id,
    TILE_READY_SIGNAL: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr
):
    
    peer_mem_tensor      = task_base_info_get_tensor(io_tensors_ptr, 0, MAX_NUM_TENSOR_DIMS)
    barrier_tensor       = task_base_info_get_tensor(io_tensors_ptr, 1, MAX_NUM_TENSOR_DIMS)
    c_tensor             = task_base_info_get_tensor(io_tensors_ptr, 2, MAX_NUM_TENSOR_DIMS)
    extra_params         = task_base_info_get_extra_params_ptr(io_tensors_ptr, 3, MAX_NUM_TENSOR_DIMS)

    B_dim = tensor_desc_size(c_tensor, 0)
    H_dim = tensor_desc_size(c_tensor, 1, 16)
    
    peer_mem_ptr = tensor_desc_data_ptr(peer_mem_tensor, tl.bfloat16)
    barrier_ptr  = tensor_desc_data_ptr(barrier_tensor, tl.int64)
    c_ptr        = tensor_desc_data_ptr(c_tensor, tl.bfloat16)
    
    rank = tl.load(extra_params).to(tl.int32)
    rank_size = tl.load(extra_params + 1).to(tl.int32)
    stride_cb = H_dim
    stride_ch = 1

    tile_id = tile_id_or_start
    
    aivIndex = sub_vec_id() 
    ncore = tl.num_programs(axis=0) # 总的 AICore 数量

    problemSize_in_rank_h = (H_dim + rank_size - 1) // rank_size
    bLoops = tl.cdiv(B_dim, BLOCK_SIZE_B)
    hLoops = tl.cdiv(problemSize_in_rank_h, BLOCK_SIZE_H)

    if aivIndex == 0:
        for r in range(tile_id, rank_size, ncore):
            dl.notify(barrier_ptr + rank * 8, r, signal=1, sig_op="set", comm_scope="intra_node")
        total_rs_blocks = bLoops * hLoops * rank_size
        p1_done_token = dl.wait(barrier_ptr, rank_size, scope="gpu", _semantic="acquire", waitValue=1)

        # =========================================================================
        # PHASE 2: All-Gather (直接拉取远程 RS Buffers)
        # =========================================================================
        for block_id in range(tile_id, total_rs_blocks, ncore):
            swizzle_b, swizzle_h = gemm_swizzle2d_Nz(
                block_id, B_dim, rank_size * BLOCK_SIZE_H * hLoops, BLOCK_SIZE_B, BLOCK_SIZE_H
            )
            src_rank_idx = swizzle_h // hLoops
            h_id_in_comm = swizzle_h % hLoops

            remote_peer_ptr = dl.symm_at(peer_mem_ptr, src_rank_idx)
            ready_base_ptr = dl.consume_token(remote_peer_ptr, p1_done_token)
            
            offs_b = swizzle_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
            offs_h_relative = h_id_in_comm * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
            offs_h_absolute = (src_rank_idx * problemSize_in_rank_h + offs_h_relative)
            
            mask_b = offs_b < B_dim
            mask_h = (offs_h_absolute < (src_rank_idx + 1) * problemSize_in_rank_h) & (offs_h_absolute < H_dim)
            mask = mask_b[:, None] & mask_h[None, :]
            
            rs_offs = offs_b[:, None] * problemSize_in_rank_h + offs_h_relative[None, :]
            gathered_val = tl.load(ready_base_ptr + rs_offs, mask=mask, other=0.0)
            
            c_offs = stride_cb * offs_b[:, None] + stride_ch * offs_h_absolute[None, :]
            tl.store(c_ptr + c_offs, gathered_val, mask=mask)

        scoreboard_release_tile_flat(
            scoreboard_ptr, 
            layer_id=layer_id, 
            task_id=task_id,
            tile_id=tile_id_or_start,
            TILE_READY_SIGNAL=TILE_READY_SIGNAL,
            MAX_TASK_ID=MAX_TASK_ID,
            MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP
        )