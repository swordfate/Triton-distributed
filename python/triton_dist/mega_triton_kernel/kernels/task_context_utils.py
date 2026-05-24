
import triton
import triton.language as tl
try:
    from triton.language.extra.cann import extension as al
except ImportError:
    al = None
    print("No al!")

# ----------------------------------------------------------------
# 展开后的 TensorDesc 功能
# ----------------------------------------------------------------

@triton.jit
def tensor_desc_data_ptr(base_ptr, dtype):
    """
    替代原 TensorDesc.data_ptr()。
    直接传入 tensor 的 base_ptr，返回其数据指针。
    """
    # buf_ptr = base_ptr.to(tl.pointer_type(tl.uint64)) # 不支持把两个uint32的指针重新解释为uint64的指针
    
    lo = tl.load(base_ptr)
    hi = tl.load(base_ptr + 1)
    val_lo = lo.to(tl.uint64) & 0xFFFFFFFF  # 确保当作无符号处理
    val_hi = hi.to(tl.uint64)
    restored_addr_val = (val_hi << 32) | val_lo
    data_ptr = restored_addr_val.to(tl.pointer_type(dtype))
    data_ptr = tl.multiple_of(data_ptr, 16)
    return data_ptr

@triton.jit
def tensor_desc_size(base_ptr, i, multiple=tl.constexpr(1)):
    """
    替代原 TensorDesc.size()。
    直接传入 tensor 的 base_ptr，返回其维度大小。
    """
    int_per_data_ptr = 2
    dim = tl.load(base_ptr + i + int_per_data_ptr)
    dim = tl.multiple_of(dim, multiple)
    dim = dim.to(tl.int32)
    return dim

# ----------------------------------------------------------------
# 展开后的 TaskBaseInfo 功能
# ----------------------------------------------------------------

@triton.jit
def task_base_info_get_tensor(io_tensors_ptr, idx, MAX_NUM_TENSOR_DIMS: tl.constexpr):
    """
    替代原 TaskBaseInfo.get_tensor()。
    注意：它不再返回一个对象，而是返回可用于 tensor_desc_* 函数的 base_ptr。
    """
    INT_PER_TENSOR = MAX_NUM_TENSOR_DIMS + 2
    return io_tensors_ptr + idx * INT_PER_TENSOR

@triton.jit
def task_base_info_get_extra_params_ptr(io_tensors_ptr, num_io_tensors, 
                                        MAX_NUM_TENSOR_DIMS: tl.constexpr):
    """
    替代原 TaskBaseInfo.get_extra_params_ptr()。
    """
    INT_PER_TENSOR = MAX_NUM_TENSOR_DIMS + 2
    return io_tensors_ptr + num_io_tensors * INT_PER_TENSOR

# ----------------------------------------------------------------
# 展开后的 Scoreboard 功能
# ----------------------------------------------------------------


@triton.jit
def scoreboard_wait_deps_flat(
    # --- Scoreboard 状态 ---
    scoreboard_table, 
    task_deps_ptr,
    
    # --- TaskBaseInfo 状态 ---
    depend_entry_start, 
    depend_entry_end,
    
    # --- 常量 ---
    INT_PER_DEPS: tl.constexpr,
    TILE_READY_SIGNAL: tl.constexpr,
    debug_counts,
):
    for t in range(depend_entry_start, depend_entry_end):
        l = tl.load(task_deps_ptr + t * INT_PER_DEPS + 0)
        r = tl.load(task_deps_ptr + t * INT_PER_DEPS + 1)
        num_signals = r - l
        sb_wait_base_ptr = scoreboard_table + l

        # for i_base in range(0, num_signals):
        #     curr_signal = tl.load(sb_wait_base_ptr + i_base, cache_modifier=".cv").to(tl.int32)
        #     count = 0
        #     while (curr_signal != TILE_READY_SIGNAL):
        #         curr_signal = tl.load(sb_wait_base_ptr + i_base, cache_modifier=".cv").to(tl.int32)
        #         count += 1
        #     if curr_signal == TILE_READY_SIGNAL:
        #         tl.store(debug_counts + l + i_base, count) # 否则上面的while代码编译后会被消除掉
        #     else:
        #         tl.store(debug_counts + l + i_base, count)
        
        for i_base in range(0, num_signals, 128):
            offsets = i_base + tl.arange(0, 128)
            mask = offsets < num_signals
            current_flag_ptrs = sb_wait_base_ptr + offsets
            
            all_ready_val = 0
            while all_ready_val == 0:
                vals = tl.load(current_flag_ptrs, mask=mask, other=TILE_READY_SIGNAL, cache_modifier=".cv", volatile=True).to(tl.int32)
                min_val = tl.min(vals, axis=0)
                if min_val == TILE_READY_SIGNAL:
                    all_ready_val = 1
    
    

@triton.jit
def scoreboard_release_tile_flat(
    # --- Scoreboard 状态 ---
    scoreboard_ptr,
    
    # --- TaskBaseInfo 状态 ---
    layer_id, 
    task_id,
    
    # --- 其他参数 ---
    tile_id,
    
    # --- 常量 ---
    TILE_READY_SIGNAL: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        sb_layer_offset = layer_id * MAX_TASK_ID * MAX_NUM_TILES_PER_OP
        sb_set_base_ptr = scoreboard_ptr + sb_layer_offset + task_id * MAX_NUM_TILES_PER_OP

        dummy = tl.arange(0, 1)
        tl.inline_asm_elementwise(
            asm="BAR.ALL",
            constraints="=l,0",
            args=[dummy],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        # tl.atomic_add(sb_set_base_ptr + tile_id, 0)
        tl.store(sb_set_base_ptr + tile_id, TILE_READY_SIGNAL)
        # tl.atomic_add(sb_set_base_ptr + tile_id, 0)
        dummy = tl.arange(0, 1)
        tl.inline_asm_elementwise(
            asm="BAR.ALL",
            constraints="=l,0",
            args=[dummy],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    
    # dummy = tl.arange(0, 1)
    # tl.inline_asm_elementwise(
    #     asm="BAR.ALL",
    #     constraints="=l,0",
    #     args=[dummy],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # # tl.atomic_add(sb_set_base_ptr + tile_id, TILE_READY_SIGNAL)
    # tl.store(sb_set_base_ptr + tile_id, TILE_READY_SIGNAL)
    # dummy = tl.arange(0, 1)
    # tl.inline_asm_elementwise(
    #     asm="BAR.ALL",
    #     constraints="=l,0",
    #     args=[dummy],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )