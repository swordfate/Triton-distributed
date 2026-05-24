
import os

# 只有当这些变量不存在时才设置
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12350" # 任意空闲端口
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import argparse
import os
import time
import torch
import triton
from mega_triton_kernel_ascend import ModelBuilder
from mega_triton_kernel_ascend.models_mk import Qwen3Model

from mega_triton_kernel_ascend.utils import get_torch_prof_ctx
from mega_triton_kernel_ascend.models import ModelConfig
from mega_triton_kernel_ascend.models import AutoTokenizer
from mega_triton_kernel_ascend.models.utils import sample_token
from mega_triton_kernel_ascend.utils import (
    initialize_distributed,
    finalize_distributed,
)

from torch_npu.contrib import transfer_to_npu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/c00946898/model-weights/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218", type=str, help="HuggingFace model name")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--backend", default="mega_kernel", type=str,
                        choices=["mega_kernel", "triton_dist_AR", "torch"], help="backend")
    parser.add_argument("--max_length", type=int, default=9216)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--profile", default=False, action="store_true", help="enable kernel level profiling")
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--intra_kernel_profile", default=False, action="store_true",
                        help="enable intra kernel profiling")
    parser.add_argument("--input_len", type=int, default=8000, help="input sequence length")
    parser.add_argument("--gen_len", type=int, default=192, help="generation length")

    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

def long_input(input_len, tokenizer):
    context = """
You are given a long technical document about distributed systems.

Section 1: Overview
A distributed system is a collection of independent computers that appears to users as a single coherent system.
The main goals are scalability, fault tolerance, availability, and performance.

Section 2: Replication
Replication improves availability by storing copies of data on multiple machines.
If one replica fails, another replica can continue serving requests.
However, replication introduces consistency challenges.

Section 3: Consensus
Consensus protocols help multiple machines agree on a single value.
Raft and Paxos are two well-known consensus protocols.
Raft is often considered easier to understand because it separates leader election, log replication, and safety.

Section 4: Sharding
Sharding splits data into partitions.
Each shard stores only part of the full dataset.
Sharding improves scalability because different machines can serve different parts of the workload.

Section 5: Caching
Caching stores frequently accessed data closer to the user or application.
A cache can reduce latency and lower backend load.
The main risk of caching is serving stale data.

Section 6: CAP Theorem
The CAP theorem says that during a network partition, a distributed system must choose between consistency and availability.
This does not mean a system can only have two of the three properties forever.
It specifically describes behavior under network partition.

Section 7: Final Notes
In practice, system designers choose tradeoffs based on product requirements.
A banking system may prefer strong consistency.
A social media feed may prefer high availability and eventual consistency.
"""

    question = """
Question:
Based on the document, why does replication improve availability, and what problem does it introduce?
Answer:
"""

    text = context * 50 + question
    tokenizer.truncation_side = "left"
    return tokenizer(
        text,
        return_tensors="pt",
        max_length=input_len,
        truncation=True,
        padding="max_length",
    ).input_ids.cuda()


if __name__ == "__main__":
    args = parse_args()
    TP_GROUP = initialize_distributed(seed=0)
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    assert args.dtype == "bfloat16"

    dtype = DTYPE_MAP[args.dtype]
    model_config = ModelConfig(model_name=args.model, max_length=args.max_length, dtype=dtype, rank=RANK,
                               world_size=WORLD_SIZE)

    builder = ModelBuilder(rank=RANK, world_size=WORLD_SIZE, local_world_size=LOCAL_WORLD_SIZE,
                           enable_profiling=args.intra_kernel_profile, target_hw='npu')
    batch_size = args.batch_size
    history = []
    ctx = get_torch_prof_ctx(args.profile)
    history = []
    user_input = "What is the capital of France?"
    history.append({"role": "user", "content": user_input})
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    print(f'prompt after tokenizer.apply_chat_template: {prompt}')
    # CHANGE 移除使用pytorch的allocator，回退使用triton-ascend中的分配
    # def alloc_fn(size, alignment, stream):
    #     return torch.empty(size, device="cuda", dtype=torch.int8)
    # triton.set_allocator(alloc_fn)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().repeat(batch_size, 1)

    # input_ids = long_input(8000, tokenizer).repeat(batch_size, 1)
    input_ids = long_input(args.input_len, tokenizer).repeat(batch_size, 1)

    print(f'input_ids.shape = {input_ids.shape}')
    gen_len = args.gen_len
    with ctx:
        if args.backend != "mega_kernel":
            # engine = Engine(model_config, temperature=args.temperature, top_p=args.top_p, verbose=True)
            # engine.backend = args.backend
            # engine.serve(input_ids=input_ids, gen_len=gen_len)
            pass
        else:
            qwen3 = Qwen3Model(batch_size, model_config, builder, build_lm_head=True)

            input_seq_len = input_ids.shape[1]
            output_ids = []
            # --- TPOT 测量变量初始化 ---
            decode_start_time = 0
            decode_tokens_count = 0
            total_sample_time = 0

            for idx in range(gen_len + input_seq_len):
                qwen3.kv_cache.inc_offset(1)
                # print(f'idx={idx} input_seq_len={input_seq_len}')
                if idx < input_seq_len:
                    next_token = input_ids[:, idx].contiguous().reshape(-1, 1)
                    logits = qwen3.mega_forwrad(next_token) # 生成之前的kvcache
                    if idx == input_seq_len - 1:
                        next_token = sample_token(logits[:, -1, :], temperature=args.temperature, top_p=args.top_p)
                        
                        torch.cuda.synchronize() 
                        decode_start_time = time.perf_counter()
                else:
                    logits = qwen3.mega_forwrad(next_token)
                    # sample 开始
                    torch.cuda.synchronize() 
                    sample_start = time.perf_counter()
                    next_token = sample_token(logits[:, -1, :], temperature=args.temperature, top_p=args.top_p)
                    # sample 结束
                    torch.cuda.synchronize() 
                    total_sample_time += (time.perf_counter() - sample_start)
                    # decode 生成的 token 数量
                    decode_tokens_count += 1

                # print(f'inference iter={idx} logits.shape = {logits.shape}')
                # print(f'logits={logits}')
                # flat_logits = logits[0].detach().cpu().flatten()
                # mask = flat_logits != 0
                # print("logits[0] None-zero Indices:", torch.nonzero(flat_logits).squeeze().cpu().numpy())
                # print("logits[0] None-zero Values:", flat_logits[mask].detach().to(torch.float32).cpu().numpy())
                if idx >= input_seq_len - 1:
                    output_ids.append(next_token)
                    next_token_cpu = next_token[0, -1].cpu()
                    # if next_token_cpu.item() == qwen3.eos_token_id:
                    #     break
            
            # 确保所有 Decode kernel 执行完毕
            torch.cuda.synchronize()
            decode_end_time = time.perf_counter()

            output_ids = torch.cat(output_ids, dim=1).cpu().tolist()
            # if RANK == 0:
            print(f"output = {tokenizer.batch_decode(output_ids, skip_special_tokens=True)}")
                # === 计算并打印 TPOT ===
            if decode_tokens_count > 0:
                total_decode_time = decode_end_time - decode_start_time
                pure_inference_time = total_decode_time - total_sample_time
                tpot = (total_decode_time / decode_tokens_count) * 1000 # 转换为毫秒
                tokens_per_sec = decode_tokens_count / total_decode_time
                
                print(f"\n{'='*20} Performance Metrics {'='*20}")
                print(f"Total Decode Time: {total_decode_time:.4f} s")
                print(f"Pure Model Time  : {pure_inference_time:.4f} s")
                print(f"Generated Tokens : {decode_tokens_count}")
                print(f"TPOT             : {tpot:.2f} ms/token")
                print(f"Throughput       : {tokens_per_sec:.2f} tokens/s")
                print(f"{'='*60}\n")
                # =====================
            torch.cuda.synchronize()
            if args.intra_kernel_profile:
                builder.dump_trace()
            # builder.finalize()
    if args.profile:
        import os
        prof_dir = f"prof/qwen3_model_{args.model.split('-')[-1]}_bs_{batch_size}_tp{WORLD_SIZE}_backend{args.backend}/"
        os.makedirs(prof_dir, exist_ok=True)
        print(f'prof_dir = {prof_dir}')
        print(f"{prof_dir}/rank_{RANK}.json")
        ctx.export_chrome_trace(f"{prof_dir}/rank_{RANK}.json")
    torch.distributed.barrier()
    finalize_distributed()
tools/profiler

import torch

# PROFILER_ENTRY_DTYPE = torch.uint64
PROFILER_ENTRY_DTYPE = torch.int64
# EMPTY_VALUE = 0xFFFFFFFFFFFFFFFF
EMPTY_VALUE = -1
# Whether to export trace. Default is True.
export_trace_on = True


def set_export_trace_on():
    global export_trace_on
    export_trace_on = True


def set_export_trace_off():
    global export_trace_on
    export_trace_on = False


def get_export_trace_on():
    global export_trace_on
    return export_trace_on


def alloc_profiler_buffer(max_num_profile_slots):
    buf = torch.empty((max_num_profile_slots, ), dtype=PROFILER_ENTRY_DTYPE, device=torch.npu.current_device())
    buf = reset_profiler_buffer(buf)
    return buf


def reset_profiler_buffer(buf):
    buf.fill_(EMPTY_VALUE)
    return buf


# def is_empty_slot(val):
#     return val == EMPTY_VALUE

def is_empty_slot(val):
    return val == -1 or val == 0xFFFFFFFFFFFFFFFF


class ProfilerBuffer:

    def __init__(self, max_num_profile_slots, trace_file, task_names):
        self.trace_file = trace_file
        self.task_names = task_names
        self.profiler_buffer = alloc_profiler_buffer(max_num_profile_slots)

    def __enter__(self):
        reset_profiler_buffer(self.profiler_buffer)
        return self.profiler_buffer

    def __exit__(self, *args, **kwargs):
        if get_export_trace_on():
            from .viewer import export_to_perfetto_trace
            export_to_perfetto_trace(self.profiler_buffer, self.task_names, self.trace_file, verbose=False)