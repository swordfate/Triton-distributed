import triton
import triton.language as tl
try:
    from triton.language.extra.cann.extension import sub_vec_id
    from triton.language.extra.cann import extension as al
except ImportError:
    ash = None
    print("No al!")
# 常量定义
NUM_BITS_ID = 20 # smid
NUM_BITS_TASK_TYPE = 11 # task type
NUM_BITS_EVENT = 1 # start / end


@triton.jit
def pack_b32_to_b64(high, low):
    """
    将两个32位整数打包成一个64位整数
    """
    # if hasattr(high, "to"):
    #     h_u64 = high.to(tl.uint64)
    # else:
    #     h_u64 = high # 保持原样 (Python int/constexpr)

    # # 2. 处理 low
    # if hasattr(low, "to"):
    #     l_u64 = low.to(tl.uint64)
    # else:
    #     l_u64 = low
    h_u64 = high
    l_u64 = low
    return (h_u64 << 32) | (l_u64 & 0xFFFFFFFF)

@triton.jit
def get_flat_pid():
    """
    获取扁平化的 Program ID
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    n_x = tl.num_programs(0)
    n_y = tl.num_programs(1)
    return pid_x + pid_y * n_x + pid_z * n_x * n_y

@triton.jit
def init_profiler(pid, profiler_buffer, group_id, num_groups, num_blocks, ENABLE_PROFILING: tl.constexpr):
    """
    初始化 Profiler：
    1. 写入 Header 信息 (Grid大小, Block映射关系)
    2. 返回当前 Block/Group 对应的日志写入起始指针
    3. 返回 stride (步长)
    """
    # 强制转换指针类型为 uint64
    ptr_base = profiler_buffer.to(tl.pointer_type(tl.uint64))
    
    # pid = get_flat_pid()
    pid_64 = pid.to(tl.int64)
    
    sm_id = pid_64

    if ENABLE_PROFILING:
        # 1. 写入全局元数据 [num_blocks, num_groups]
        if pid_64 == 0:
            h_tensor = tl.full((1,), num_blocks, dtype=tl.uint64)
            l_tensor = tl.full((1,), num_groups, dtype=tl.uint64)
            global_meta = pack_b32_to_b64(h_tensor, l_tensor)
            tl.store(ptr_base + tl.arange(0,1), global_meta)

        # 2. 写入 Block 元数据 [pid, sm_id]
        # 偏移量是 1 + pid
        block_meta = pack_b32_to_b64(pid_64, sm_id)
        block_offset = 1 + pid_64
        tl.store(ptr_base + block_offset, block_meta)

    # 3. 计算日志区的起始地址
    # 跳过 Header 区域 (大小为 1 + num_blocks)
    header_offset = 1 + num_blocks
    # 计算当前 Group 的列偏移
    my_offset = pid_64 * num_groups + group_id
    # 初始指针位置
    total_start_offset = header_offset + my_offset
    
    # 计算步长 stride
    stride = num_blocks * num_groups
    
    return ptr_base, total_start_offset, stride

# @triton.jit
# def get_sys_cnt(dummy_input, dep):
#     dep_vec = dummy_input + dep.to(tl.int64)

#     return tl.inline_asm_elementwise(
#         asm="""
#         MOV $0, $1
#         MOV $0, SYS_CNT
#         """,
#         constraints="=l,l",
#         args=[dep_vec],
#         dtype=tl.int64,
#         is_pure=False,
#         pack=1,
#     )

# @triton.jit
# def get_sys_cnt(dummy_input):

#     t0 = tl.full((1,), 0, tl.int64)
#     t1 = tl.full((1,), 0, tl.int64)

#     for i in al.parallel(0, 2, bind_sub_block=True):
#         time_stamp = tl.inline_asm_elementwise(
#             asm="MOV $0, SYS_CNT",
#             constraints="=l,l",
#             args=[dummy_input],
#             dtype=tl.int64,
#             is_pure=False,
#             pack=1,
#         )

#         if i == 0:
#             t0 = time_stamp
#         else:
#             t1 = time_stamp

#     return tl.minimum(t0, t1)

@triton.jit
def get_sys_cnt(dummy_input):
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT",
        constraints="=l,l",
        args=[dummy_input],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )

@triton.jit
def record_event(base_ptr, current_offset, stride, pid, group_id, num_groups, is_start, task_type, ENABLE_PROFILING: tl.constexpr):
    """
    记录事件：
    1. 获取时间戳
    2. 编码 Tag
    3. 写入显存
    4. 返回下一个写入位置的指针
    """
    if ENABLE_PROFILING:
        i = al.sub_vec_id()
        if i == 0:
            # 1. 获取时间戳 (替换 globaltimer_lo)
            offsets = tl.arange(0, 1)
            dummy = offsets.to(tl.int64)
            # TODO：may incure bugs
            # t0, t1 = get_sys_cnt(dummy)
            # timestamp_64 = (t1 - t0) * 20
            timestamp_64 = get_sys_cnt(dummy) * 20
            
            # 2. 编码 Tag
            pid_64 = pid.to(tl.int64)
            global_id = pid_64 * num_groups + group_id
            
            # 拼接 Tag: [GlobalID | TaskType | IsStart]
            # 偏移量计算 1+11+20
            GLOBAL_ID_OFFSET: tl.constexpr = 11 + 1
            TASK_TYPE_OFFSET: tl.constexpr = 1
            tag = (global_id << GLOBAL_ID_OFFSET) | (task_type << TASK_TYPE_OFFSET) | is_start
            
            # 3. 打包成 64位 Entry [Tag | Timestamp]
            # tag_u64 = tl.full((1,), tag, dtype=tl.uint64)
            tag_vec = tag + tl.zeros((1,), dtype=tl.int64)
            tag_u64 = tag_vec.to(tl.uint64)

            entry = pack_b32_to_b64(tag_u64, timestamp_64)
            # 4. 写入显存
            target_ptr = base_ptr + current_offset + tl.arange(0, 1)
            tl.store(target_ptr, entry)

        return current_offset + stride
    else:
        return current_offset