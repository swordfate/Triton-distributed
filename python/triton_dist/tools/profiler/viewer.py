from typing import Any, Dict, List, Tuple
import numpy as np
from .language_npu import (
    NUM_BITS_ID,
    NUM_BITS_TASK_TYPE,
    NUM_BITS_EVENT,
)
from .context import is_empty_slot

import torch
from dataclasses import dataclass


# adapt from flashinfer/flashinfer/profiler/__init__.py
def decode_tag(tag, num_groups):
    """
    Decode a profiler tag into (block_idx, group_idx, task_type, is_start).
    Tag layout:  GLOBAL_ID | TASK TYPE | IS START
    """
    global_id = (tag >> (NUM_BITS_TASK_TYPE + NUM_BITS_EVENT)) & ((1 << NUM_BITS_ID) - 1)
    task_type = (tag >> NUM_BITS_EVENT) & ((1 << NUM_BITS_TASK_TYPE) - 1)
    is_start = tag & NUM_BITS_EVENT
    block_idx = global_id // num_groups
    group_idx = global_id % num_groups
    assert NUM_BITS_EVENT == 1
    return block_idx, group_idx, task_type, is_start


# adapt from flashinfer/flashinfer/profiler/__init__.py
def export_to_perfetto_trace(profiler_buffer: torch.Tensor, task_names: List[str], file_name: str,
                             verbose: bool = False) -> None:
    from tg4perfetto import TraceGenerator

    if not file_name.endswith(".perfetto-trace"):
        file_name = file_name + ".perfetto-trace"
    # assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    # print(f'profiler_buffer_host[:25] = {profiler_buffer_host[:25]}')
    num_groups, num_blocks = profiler_buffer_host[:1].view(dtype=torch.uint32) # little end
    # print(f'num_blocks = {num_blocks}, num_groups = {num_groups}')
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)
    
    # 新增：用于记录区间的列表
    wait_intervals = []
    compute_intervals = []
    # 记录全局最早和最晚时间戳，用于计算总比例
    min_ts = float('inf')
    max_ts = float('-inf')

    tgen = TraceGenerator(file_name)

    pid_map = {}
    track_map: Dict[Tuple[int, int, int], Any] = {}
    begin_timestamp_map = {}

    block_idx_to_smid = {}
    profiler_buffer_host = profiler_buffer_host[1:]
    # for better view
    if num_groups == 1:
        pid_master = tgen.create_group("tracks of all blocks")

    for i in range(num_blocks):
        sm_id, block_idx = profiler_buffer_host[i:i + 1].view(dtype=torch.uint32) # little end
        block_idx, sm_id = int(block_idx), int(sm_id)
        block_idx_to_smid[block_idx] = sm_id
        if num_groups > 1:
            if block_idx not in pid_map:
                pid_map[block_idx] = tgen.create_group(f"block_{block_idx}_sm_{sm_id}")
        else:
            pid_map[block_idx] = pid_master
    if verbose:
        print(f"block_idx_to_smid = {block_idx_to_smid}, {len(block_idx_to_smid)}")

    profiler_buffer_host = profiler_buffer_host[num_blocks:].numpy()
    empty_count = 0
    for i in range(len(profiler_buffer_host)):
        if is_empty_slot(profiler_buffer_host[i]):
            empty_count += 1
            if empty_count > num_blocks * num_groups:
                break
            continue
        empty_count = 0
        timestamp, tag = profiler_buffer_host[i:i + 1].view(np.uint32) # little end
        tag = int(tag)
        timestamp = int(timestamp)
        # print(f'tag = {tag}, timestamp = {timestamp}')
        block_idx, group_idx, task_type, is_start = decode_tag(tag, num_groups)
        sm_id = block_idx_to_smid[block_idx]
        if verbose:
            print(
                f'tag = {tag}, block_idx = {block_idx}, task_type = {task_type}, name = {task_names[task_type]} is_start = {is_start}, timestamp = {timestamp}'
            )
        # create trackers
        pid = pid_map[block_idx]
        cur_task_name = task_names[task_type]
        
        # 更新全局时间跨度
        min_ts = min(min_ts, timestamp)
        max_ts = max(max_ts, timestamp)

        if (block_idx, group_idx, task_type) in track_map:
            track = track_map[(block_idx, group_idx, task_type)]
        else:
            # assert is_start
            if num_groups > 1:
                track = pid.create_track(f"group_{group_idx}")
            else:
                track = pid.create_track(f"block_{block_idx}_sm_{sm_id}_group_{group_idx}")
            track_map[(block_idx, group_idx, task_type)] = track

        if is_start:
            track.open(timestamp, cur_task_name)
            begin_timestamp_map[(block_idx, group_idx, task_type)] = timestamp
        else:
            track.close(timestamp)
            begin_timestamp = begin_timestamp_map[(block_idx, group_idx, task_type)]
            # if begin_timestamp > timestamp:
            #     print("[bad pair]")
            #     print(f"key={(block_idx, group_idx, task_type)}")
            #     print(f"start={begin_timestamp}, end={timestamp}, name={task_names[task_type]}")

            #     lo = max(0, i - 20)
            #     hi = min(len(profiler_buffer_host), i + 20)

            #     for j in range(lo, hi):
            #         raw = int(profiler_buffer_host[j])
            #         if is_empty_slot(raw):
            #             print(j, "EMPTY")
            #             continue

            #         ts, tg = np.array([raw], dtype=np.uint64).view(np.uint32)
            #         b, g, t, s = decode_tag(int(tg), num_groups)
            #         print(f"j={j}, tag={int(tg)}, ts={int(ts)}, block={b}, group={g}, task={t}, start={s}")
            assert begin_timestamp <= timestamp, f"timestamp overflow, start = {begin_timestamp}, end = {timestamp}, tasktype = {task_type}, name = {task_names[task_type]}, block_idx = {block_idx}, group_idx = {group_idx}"
            if begin_timestamp is not None:
                interval = (begin_timestamp, timestamp)
                # 判定为 wait 事件
                if "scoreboard_wait_deps" in cur_task_name.lower():
                    wait_intervals.append(interval)
                else:
                    compute_intervals.append(interval)
    # --- 循环结束后进行统计 ---
    pure_wait_time = calculate_pure_wait(wait_intervals, compute_intervals)
    total_span = max_ts - min_ts if max_ts > min_ts else 0
    
    if total_span > 0:
        pure_wait_ratio = (pure_wait_time / total_span) * 100
        print(f"\n[Profiler Statistics]")
        print(f"Total Execution Time: {total_span*0.001*0.001} ms")
        print(f"Pure Wait Duration:   {pure_wait_time*0.001*0.001} ms")
        print(f"Pure Wait Ratio:      {pure_wait_ratio:.2f}%")
        
    tgen.flush()
    

def calculate_pure_wait(wait_intervals, compute_intervals):
    """计算纯 Wait 时间：Union(wait) - Union(compute)"""
    
    def merge_intervals(intervals):
        if not intervals: return []
        intervals.sort(key=lambda x: x[0])
        merged = []
        curr_start, curr_end = intervals[0]
        for n_start, n_end in intervals[1:]:
            if n_start <= curr_end:
                curr_end = max(curr_end, n_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = n_start, n_end
        merged.append((curr_start, curr_end))
        return merged

    # 1. 分别求并集
    m_wait = merge_intervals(wait_intervals)
    m_compute = merge_intervals(compute_intervals)

    # 2. 从 wait 并集中减去 compute 并集
    pure_wait_duration = 0
    pure_wait_collections = []
    for ws, we in m_wait:
        # 初始认为整个 wait 区间都是纯的
        current_segments = [(ws, we)]
        for cs, ce in m_compute:
            next_segments = []
            for ss, se in current_segments:
                # 如果计算区间和当前 wait 片段有重叠
                if cs < se and ce > ss:
                    # 左边留下的部分
                    if cs > ss:
                        next_segments.append((ss, cs))
                    # 右边留下的部分
                    if ce < se:
                        next_segments.append((ce, se))
                else:
                    # 没重叠，原样保留
                    next_segments.append((ss, se))
            current_segments = next_segments
        pure_wait_collections.extend([(e-s,s,e) for s,e in current_segments])
        pure_wait_duration += sum(s[1] - s[0] for s in current_segments)
    
    print(f'pure_wait_time_intervals len= {len(pure_wait_collections)}')
    print(f'pure_wait_time_intervals (ns) = {[round(during,1) for during, s, e in pure_wait_collections]}')
    return pure_wait_duration


@dataclass
class Task:
    tag: int
    task_type: int
    start_time: int  # ns
    duration: int  # ns


def parse_to_tracks(profiler_buffer: torch.Tensor):
    assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    begin_timestamp_map = {}

    block_idx_to_smid = {}
    profiler_buffer_host = profiler_buffer_host[1:]
    block_idx_to_tracks = {}
    for i in range(num_blocks):
        block_idx, sm_id = profiler_buffer_host[i:i + 1].view(dtype=torch.uint32)
        block_idx, sm_id = int(block_idx), int(sm_id)
        block_idx_to_smid[block_idx] = sm_id
        block_idx_to_tracks[block_idx] = []

    profiler_buffer_host = profiler_buffer_host[num_blocks:].numpy()

    empty_count = 0
    for i in range(len(profiler_buffer_host)):
        if is_empty_slot(profiler_buffer_host[i]):
            empty_count += 1
            if empty_count > num_blocks * num_groups:
                break
            continue
        empty_count = 0
        tag, timestamp = profiler_buffer_host[i:i + 1].view(np.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, task_type, is_start = decode_tag(tag, num_groups)
        sm_id = block_idx_to_smid[block_idx]

        if is_start:
            begin_timestamp_map[(block_idx, group_idx, task_type)] = timestamp
        else:
            begin_timestamp = begin_timestamp_map[(block_idx, group_idx, task_type)]
            assert begin_timestamp < timestamp, f"timestamp overflow, start = {begin_timestamp}, end = {timestamp}, tasktype = {task_type}, block_idx = {block_idx}, group_idx = {group_idx}"
            track = Task(tag=tag, task_type=task_type, start_time=begin_timestamp, duration=timestamp - begin_timestamp)
            block_idx_to_tracks[block_idx].append(track)
    return block_idx_to_tracks