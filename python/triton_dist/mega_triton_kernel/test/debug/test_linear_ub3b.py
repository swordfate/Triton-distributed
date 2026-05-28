"""同 test_linear_ub3.py, 但打印完整错误信息"""
import torch, triton, triton.language as tl, sys
from torch_npu.contrib import transfer_to_npu


@triton.jit
def kernel_with_wrapper(
    work_queues, num_tasks_per_wq,
    INT_PER_TASK: tl.constexpr, NUM_SMS: tl.constexpr, MAX_NUM_TENSOR_DIMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    SUB_BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, NUM_STAGES: tl.constexpr,
    scoreboard_ptr, task_deps_ptr,
    INT_PER_DEPS: tl.constexpr, MAX_TASK_ID: tl.constexpr, MAX_NUM_TILES_PER_OP: tl.constexpr,
    debug_counts,
):
    sm_id = tl.program_id(axis=0)
    num_tasks = tl.load(num_tasks_per_wq + sm_id)
    offset = INT_PER_TASK * NUM_SMS
    TASK_TYPE_OFFSET,LAYER_ID_OFFSET,TASK_ID_OFFSET = 0,1,2
    TILE_ID_OR_START_OFFSET,DEPEND_ENTRY_START_OFFSET,DEPEND_ENTRY_END_OFFSET = 3,4,5
    IO_TENSORS_OFFSET = 6
    for i in range(num_tasks):
        task_type = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + TASK_TYPE_OFFSET).to(tl.int32)
        layer_id = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + LAYER_ID_OFFSET).to(tl.int32)
        task_id = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + TASK_ID_OFFSET).to(tl.int32)
        tile_id_or_start = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + TILE_ID_OR_START_OFFSET).to(tl.int32)
        depend_entry_start = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + DEPEND_ENTRY_START_OFFSET).to(tl.int32)
        depend_entry_end = tl.load(work_queues + i*offset + sm_id*INT_PER_TASK + DEPEND_ENTRY_END_OFFSET).to(tl.int32)
        io_tensors_ptr = work_queues + i*offset + sm_id*INT_PER_TASK + IO_TENSORS_OFFSET

        if task_type == 1:
            linear_task_compute_lite(
                tile_id_or_start, MAX_NUM_TENSOR_DIMS, io_tensors_ptr,
                BLOCK_SIZE_M, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES,
                scoreboard_ptr, layer_id, task_id,
                TILE_READY_SIGNAL=tl.constexpr(2), MAX_TASK_ID=MAX_TASK_ID, MAX_NUM_TILES_PER_OP=MAX_NUM_TILES_PER_OP,
            )


@triton.jit
def linear_task_compute_lite(
    tile_id, MD, io_ptr,
    BM: tl.constexpr, BN: tl.constexpr, SUB: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr,
    sb, lid, tid, TILE_READY_SIGNAL: tl.constexpr, MTI: tl.constexpr, MT: tl.constexpr,
):
    IPT = MD + 2
    al=io_ptr;ah=io_ptr+1;bl=io_ptr+IPT;bh=io_ptr+IPT+1;cl=io_ptr+2*IPT;ch=io_ptr+2*IPT+1
    a_ptr=((tl.load(ah).to(tl.uint64)<<32)|(tl.load(al).to(tl.uint64)&0xFFFFFFFF)).to(tl.pointer_type(tl.bfloat16))
    b_ptr=((tl.load(bh).to(tl.uint64)<<32)|(tl.load(bl).to(tl.uint64)&0xFFFFFFFF)).to(tl.pointer_type(tl.bfloat16))
    c_ptr=((tl.load(ch).to(tl.uint64)<<32)|(tl.load(cl).to(tl.uint64)&0xFFFFFFFF)).to(tl.pointer_type(tl.bfloat16))
    M=tl.load(io_ptr+2).to(tl.int32);K=tl.load(io_ptr+3).to(tl.int32);N_c=tl.load(io_ptr+2*IPT+3).to(tl.int32)
    npn=tl.cdiv(N_c,BN);pm=tile_id//npn;pn=tile_id%npn;sm=pm*BM;bn=pn*BN;oa=sm+tl.arange(0,BM);kt=tl.cdiv(K,BK)
    for i in tl.range(0,BN,SUB,num_stages=NS):
        sn=bn+i;ob=sn+tl.arange(0,SUB);acc=tl.zeros((BM,SUB),dtype=tl.float32)
        for ki in range(kt):
            ok=ki*BK+tl.arange(0,BK);ap=a_ptr+(oa[:,None]*K+ok[None,:]);bp=b_ptr+(ob[:,None]*K+ok[None,:])
            a=tl.load(ap);b=tl.load(bp);acc=tl.dot(a,b.T,acc)
        oc=pm*BM+tl.arange(0,BM);cn=sn+tl.arange(0,SUB);cp=c_ptr+N_c*oc[:,None]+cn[None,:];tl.store(cp,acc.to(tl.bfloat16))


def make_wq(device):
    M,N,K=1,6144,4096
    a=torch.randn(M,K,dtype=torch.bfloat16,device=device)
    b=torch.randn(N,K,dtype=torch.bfloat16,device=device)
    c=torch.zeros(M,N,dtype=torch.bfloat16,device=device)
    IP=6;IPT=6+3*IP;wq=torch.zeros(IPT,dtype=torch.int32,device=device)
    wq[0]=1
    for idx,t in enumerate([a,b,c]):
        base=6+idx*IP;ptr=t.data_ptr()
        wq[base]=ptr&0xFFFFFFFF;wq[base+1]=(ptr>>32)&0xFFFFFFFF
        shape=list(t.shape)+[1]*(4-len(t.shape))
        for d in range(4):wq[base+2+d]=shape[d]
    return wq.reshape(1,1,IPT),a,b,c


if __name__=="__main__":
    device='npu'
    print(f"=== Wrapper test ns5 ===")
    wq,a,b,c=make_wq(device);nt=torch.tensor([1],dtype=torch.int32,device=device)
    sb=torch.zeros(100,dtype=torch.int32,device=device)
    td=torch.zeros(0,dtype=torch.int32,device=device).reshape(0,2)
    dc=torch.zeros(10,dtype=torch.int32,device=device)
    try:
        kernel_with_wrapper[(1,1,1)](
            wq,nt,INT_PER_TASK=wq.shape[2],NUM_SMS=1,MAX_NUM_TENSOR_DIMS=4,
            BLOCK_SIZE_M=16,BLOCK_SIZE_N=6240,SUB_BLOCK_SIZE_N=416,BLOCK_SIZE_K=256,NUM_STAGES=5,
            scoreboard_ptr=sb,task_deps_ptr=td,INT_PER_DEPS=2,MAX_TASK_ID=10,MAX_NUM_TILES_PER_OP=128,
            debug_counts=dc,
        )
        torch.npu.synchronize()
        print(f"OK c_sum={c.sum().item():.1f}")
    except Exception as e:
        print(f"FULL ERROR:\n{e}")
