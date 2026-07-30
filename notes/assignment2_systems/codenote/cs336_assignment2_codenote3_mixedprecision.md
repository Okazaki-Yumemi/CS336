修改了benchmark内的函数

```py

def run_step(
    model : BasicsTransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    optimizer : AdamW,
    precision: str
):
    if mode != "forward":
        optimizer.zero_grad(set_to_none= True)
        
    use_bf16 = precision == "bf16"
        
    with torch.autocast(
        device_type= "cuda",
        dtype= torch.bfloat16,
        enabled= use_bf16,
    ):
        logits = model(inputs)
        
        if mode != "forward":
            loss = cross_entropy(logits,targets)
    
    if mode == "forward":
        return
    
    loss.backward()
    
    if mode == "train":
        optimizer.step()
```







```bash
for precision in fp32 bf16; do
  for mode in forward backward; do
    uv run python scripts/benchmark.py \
      --model-size small \
      --context-length 512 \
      --mode "$mode" \
      --precision "$precision" \
      --warmup-steps 5 \
      --measurement-steps 10
  done
done

=========================================
device: cuda
model size: small
mode: forward
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
precision: fp32
mean time(ms): 76.31485929941846
std(ms) : 0.15108773613247733
=========================================
=========================================
device: cuda
model size: small
mode: backward
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
precision: fp32
mean time(ms): 231.04905740001414
std(ms) : 0.34227912048477116
=========================================
=========================================
device: cuda
model size: small
mode: forward
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
precision: bf16
mean time(ms): 43.252240699803224
std(ms) : 0.11118565662202241
=========================================
=========================================
device: cuda
model size: small
mode: backward
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
precision: bf16
mean time(ms): 130.83812260028935
std(ms) : 0.0682033727200203
=========================================
```

| Precision |   Forward | Forward + Backward |
| --------- | --------: | -----------------: |
| FP32      | 76.315 ms |         231.049 ms |
| BF16      | 43.252 ms |         130.838 ms |


加速比 forward 1.76x, forward + backward 1.77x

