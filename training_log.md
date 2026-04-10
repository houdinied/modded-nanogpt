# Training Log

## Hardware

- **Current machine**: 8x NVIDIA B200 (183 GB HBM3e each), CUDA driver 580.126.09

---

## Run 1: llm.c GPT-2 124M (H100)

| Field | Value |
|---|---|
| **Framework** | llm.c (CUDA C) |
| **GPU** | 8x H100 80GB SXM |
| **Model** | GPT-2 124M (`d12`: 12 layers, 12 heads, 768 embd) |
| **Dataset** | FineWeb 10B |
| **Precision** | BF16 (cuDNN FlashAttention) |
| **Optimizer** | AdamW (lr=0.0006, warmup=700, wd=0.1) |
| **Batch** | 64 seq/gpu x 1024 tok x 8 gpu = 524,288 tok/step |
| **Steps** | 19,645 (target 18,865) |
| **Total time** | ~135 min (run1: 107.6 min, run2 resumed from ckpt 15000: 27.4 min) |
| **Avg speed** | ~354 ms/step |
| **Final val loss** | 3.2703 |
| **HellaSwag** | 0.3039 |

---

## Run 2: llm.c GPT-2 124M (B200)

| Field | Value |
|---|---|
| **Framework** | llm.c (CUDA C) |
| **GPU** | 8x B200 183GB |
| **Model** | GPT-2 124M (`d12`: 12 layers, 12 heads, 768 embd) |
| **Dataset** | FineWeb 10B |
| **Precision** | BF16 (cuDNN FlashAttention) |
| **Optimizer** | AdamW (lr=0.0006, warmup=700, wd=0.1) |
| **Batch** | 64 seq/gpu x 1024 tok x 8 gpu = 524,288 tok/step |
| **Steps** | 19,645 |
| **Final val loss** | 3.2708 |
| **HellaSwag** | 0.3021 |
| **Notes** | Same hyperparams as H100 run. No Blackwell-specific opts (no FP8/FP4). |

---

## Run 3: modded-nanogpt GPT-2 124M (B200) - Unoptimized Baseline

| Field | Value |
|---|---|
| **Framework** | modded-nanogpt (PyTorch, torch.compile) |
| **GPU** | 8x B200 183GB |
| **Model** | GPT-2 124M (`d12`: 12 layers, 12 heads, 768 embd) |
| **Dataset** | FineWeb 10B |
| **Precision** | BF16 (autocast) |
| **Optimizer** | AdamW (lr=0.0018, betas=0.9/0.95, wd=0.1) |
| **Batch** | 64 seq/gpu x 1024 tok x 8 gpu = 524,288 tok/step |
| **Steps** | 12,288 (warmup=256, warmdown=2048) |
| **Architecture mods** | RoPE, RMSNorm, no bias, weight tying, attention scaling (1/sqrt(2*n_layer)) |
| **Total time** | ~49.5 min |
| **Avg speed (last 19)** | 260.7 ms/step |
| **Throughput** | ~2.5M tok/s |
| **Peak memory** | 37,069 MiB / 183,359 MiB (20%) |
| **Final val loss** | 3.2316 |
| **Runs** | 5 completed full runs, val losses: 3.2360, 3.2339, 3.2328, 3.2316, 3.2329 |
| **Notes** | Vanilla PyTorch DDP + torch.compile. No fused kernels, no FP8, no Muon. Model too small to saturate B200 compute (~11% MFU). Similar wall-clock to H100 despite 2.3x more FLOPS -- bottlenecked by kernel launch overhead, DDP latency, and memory-bandwidth-bound ops at this model size. |
