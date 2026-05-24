## 运行
```bash
export PATH=triton-megakernels-ascend/src/mega_triton_kernel_ascend/test/lib:$PATH
which bishengir-compile hivmc # 验证路径在上面添加的lib目录下
ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29480 test_qwen3.py --batch_size=1 --input_len=8000 --gen_len=192
```