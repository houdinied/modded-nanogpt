# Training Log

## Summary

goal: train GPT-2 124M in FP4 on 8xB200 through the power of friendship. target val loss <=3.28.

FP4 on all layers: NaN at step 4 every time (runs 4-6). changed gradient handling, changed LR, nothing helped. then got it to not NaN but loss was completely flat at 10.96 for 800+ steps (run 7). model was not learning at all. turns out FP4 forward on attention is the problem. the dot product between Q and K squares the quantization error, then softmax exponentiates it. at 124M scale with head_dim=64, a single FP4 block of 16 covers 25% of each attention head. too much of the representation gets corrupted. quartet-II paper doesnt have this problem because they run on LLaMA 1B+ where head_dim=128 and there are way more heads to compensate.

what actually works: FP4 on MLP layers + attention output projection + lm_head, keep QKV projection in BF16. run 9 is best result: 3.2822 val loss, 148ms/step, 1.76x faster than BF16 baseline (261ms). 0.002 away from target.

also tried hot-swapping QKV from BF16 to FP4 at step 500 after weights grow past near-zero init. training worked (no NaN, loss decreased) but val loss was worse (3.2954) and torch.compile didnt recompile properly after the module swap so it ran slower than baseline. not useful.

LR tuning did nothing. the 0.002 gap is just FP4 forward quantization noise on MLP. hyperparameters cant fix that.

**next steps:** add Muon optimizer (code already exists in muon_addition_back/). Muon beats AdamW on this task independent of precision. should close the gap.

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
| **Model** | GPT-2 124M (`d12`: 12 layers, 12 heads, 768 embd) |
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
| **Notes** | Vanilla PyTorch DDP + torch.compile. No fused kernels, no FP8, no Muon. |

---

## Run 4: modded-nanogpt GPT-2 124M (B200) - Quartet-II NVFP4 (NaN divergence)

| Field | Value |
|---|---|
| **Model** | GPT-2 124M (`d12`: 12 layers, 12 heads, 768 embd, vocab padded 50257->50304) |
| **Precision** | NVFP4 forward + NVFP4 backward (Quartet-II) on all 50 linear layers, BF16 elsewhere |
| **Optimizer** | AdamW (lr=0.0018, betas=0.9/0.95, wd=0.1) |
| **Batch** | 64 seq/gpu x 1024 tok x 8 gpu = 524,288 tok/step |
| **Steps** | 3 (NaN at step 4) |
| **Warmup** | 256 steps |
| **Avg speed** | ~193 ms/step, ~2.7M tok/s |
| **Result** | NaN at step 4. Training diverged immediately. |
| **Notes** | Quartet-II NVFP4 kernels (flashinfer mm_fp4, EDEN rounding, Hadamard rotation) on all attention + MLP linears + lm_head. Vocab padded to 50304 for 128-alignment. Per-parameter gradient unit-norm too aggressive with FP4's noisier gradients. |

---

## Run 5: NVFP4 + halved LR (NaN divergence)

| Field | Value |
|---|---|
| **Change from Run 4** | lr=0.0009 (halved), warmup=512 (doubled) |
| **Steps** | 3 (NaN at step 4 again) |
| **Result** | NaN at step 4. Same failure point — LR not the issue. |
| **Notes** | Per-parameter gradient unit-norm (`p.grad / p.grad.norm()`) is the root cause, not the LR. A single NaN/inf from FP4 stochastic rounding poisons the norm, which propagates to all parameters. Replacing with `clip_grad_norm_(max_norm=1.0)` and restoring original lr=0.0018, warmup=256. |

---

## Run 6: NVFP4 + clip_grad_norm_ (NaN divergence)

| Field | Value |
|---|---|
| **Change from Run 4** | Replaced per-param unit-norm with `clip_grad_norm_(max_norm=1.0)`. Restored lr=0.0018, warmup=256. |
| **Steps** | 3 (NaN at step 4 again) |
| **Result** | NaN at step 4. Same failure point. |
| **Notes** | NaN persists regardless of gradient handling — problem is in the forward pass, not backward. Added per-layer NaN detection (embedding, each block, rmsnorm, lm_head, loss) to isolate source. |

---

## Run 7: NVFP4 forward + BF16 backward (flat loss)

| Field | Value |
|---|---|
| **Change from Run 6** | Added `disable_backward_quant=True` to all Quartet_II_linear layers. Backward now dequantizes saved FP4 to BF16 and does full-precision matmuls. NaN debug prints added. |
| **Steps** | 823 (killed manually) |
| **Avg speed** | ~187 ms/step, ~2.8M tok/s |
| **Result** | No NaN. Loss flat at ~10.96 for 800+ steps at full LR (1.8e-3). No learning. |
| **Notes** | BF16 backward didn't help — the FP4 forward itself is too lossy. At 124M scale with near-zero init weights, FP4 block scaling maps everything to the smallest representable values, destroying gradient signal. The FP4 forward quantization is the bottleneck, not the backward. Need to either use BF16/FP8 forward or apply FP4 selectively (e.g., only MLP, not attention). |

---

## Run 8: NVFP4 on MLP only, BF16 on attention (completed)

| Field | Value |
|---|---|
| **Model** | GPT-2 124M (`d12`: 12 layers, 12 heads, 768 embd, vocab padded 50257->50304) |
| **Precision** | NVFP4 forward + BF16 backward on MLP (c_fc, c_proj) + lm_head (26 layers). BF16 on attention (c_attn, c_proj). |
| **Optimizer** | AdamW (lr=0.0018, betas=0.9/0.95, wd=0.1), clip_grad_norm_(max_norm=1.0) |
| **Batch** | 64 seq/gpu x 1024 tok x 8 gpu = 524,288 tok/step |
| **Steps** | 12,288 (warmup=256, warmdown=2048) |
| **Total time** | 31.40 min |
| **Avg speed (last 19)** | 188.3 ms/step |
| **Throughput** | ~4.25M tok/s |
| **Peak memory** | 52,724 MiB |
| **Final val loss** | 3.2797 |
| **Notes** | First successful NVFP4 training run. FP4 on attention caused flat loss (Run 7) — attention QKV dot products + softmax amplify quantization error. MLP tolerates FP4 due to GELU dampening and residual dilution. 1.4x faster than BF16 baseline (188ms vs 261ms). Val loss 3.28 vs baseline 3.23 — small degradation from FP4 on MLP + lm_head. |

---

## Run 9: NVFP4 on MLP + attention c_proj + lm_head, BF16 on c_attn only (completed)

| Field | Value |
|---|---|
| **Model** | GPT-2 124M (`d12`: 12 layers, 12 heads, 768 embd, vocab padded 50257->50304) |
| **Precision** | NVFP4 forward + BF16 backward on 38/50 linear layers (MLP c_fc, MLP c_proj, attn c_proj, lm_head). BF16 on 12 c_attn (QKV) only. |
| **Optimizer** | AdamW (lr=0.0018, betas=0.9/0.95, wd=0.1), clip_grad_norm_(max_norm=1.0) |
| **Batch** | 64 seq/gpu x 1024 tok x 8 gpu = 524,288 tok/step |
| **Steps** | 12,288 (warmup=256, warmdown=2048) |
| **Total time** | 33.43 min |
| **Avg speed (last 19)** | 148.3 ms/step |
| **Throughput** | ~4.1M tok/s |
| **Peak memory** | 53,031 MiB |
| **Final val loss** | 3.2822 |
| **Notes** | Attention c_proj tolerates FP4 — only QKV (c_attn) needs BF16. Val loss 3.28, nearly same as Run 8 (3.28) with 12 more layers in FP4. 1.76x faster than BF16 baseline (148ms vs 261ms). Loss gap vs baseline (3.23) is ~0.05, likely from FP4 forward quantization noise on MLP + lm_head. |

---

## Run 10: Delayed FP4 on c_attn — all 50 layers NVFP4 after step 500 (completed)

| Field | Value |
|---|---|
| **Model** | GPT-2 124M (`d12`: 12 layers, 12 heads, 768 embd, vocab padded 50257->50304) |
| **Precision** | Steps 0-499: 38/50 FP4 (c_attn in BF16). Step 500: hot-swap c_attn to NVFP4 → all 50 linear layers FP4 forward + BF16 backward. |
| **Optimizer** | AdamW (lr=0.0018, betas=0.9/0.95, wd=0.1), clip_grad_norm_(max_norm=1.0). Optimizer rebuilt at step 500. |
| **Batch** | 64 seq/gpu x 1024 tok x 8 gpu = 524,288 tok/step |
| **Steps** | 12,288 (warmup=256, warmdown=2048, fp4_start_step=500) |
| **Total time** | 36.34 min |
| **Avg speed (last 19)** | 170.5 ms/step |
| **Throughput** | ~3.65M tok/s |
| **Peak memory** | 53,031 MiB |
| **Final val loss** | 3.2954 |
| **Notes** | Delayed FP4 on c_attn works — no NaN, no flat loss. All 50 layers in NVFP4 for 96% of training (steps 500-12288). Val loss 3.30 vs Run 9's 3.28 — small regression from FP4 QKV quantization noise. Slower avg than Run 9 (170ms vs 148ms) due to recompilation overhead at step 500 and BF16 c_attn for first 500 steps. Target of <=3.28 not met. |

---

## Run 11: Delayed FP4 + lower LR (completed)

| Field | Value |
|---|---|
| **Change from Run 10** | lr=0.0015 (down from 0.0018). Removed NaN debug prints. |
| **Steps** | 12,288 (warmup=256, warmdown=2048, fp4_start_step=500) |
| **Total time** | 44.67 min |
| **Avg speed (last 19)** | 277.9 ms/step |
| **Peak memory** | 26,950 MiB |
| **Final val loss** | 3.2954 |
| **Notes** | Same val loss as Run 10 (3.2954) — lower LR didn't help. Avg speed much slower (278ms vs 170ms) and memory halved (27GB vs 53GB), suggesting torch.compile behaved differently after the swap (possibly fell back to eager on some ops). The recompilation at step 500 may not be producing optimized kernels. Target <=3.28 not met. |


