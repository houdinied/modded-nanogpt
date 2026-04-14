# modded-nanogpt with NVFP4

GPT-2 124M training with [Quartet-II](https://arxiv.org/abs/2601.22813) NVFP4 kernels on 8xB200.

FP4 forward on MLP + attention output projection + lm_head. BF16 on QKV projection (attention is too sensitive to 4-bit quantization at 124M scale). Backward is BF16 everywhere.

Best result: **3.2822 val loss, 148ms/step, 1.76x faster than BF16 baseline**.

## Setup

```bash
docker build -t modded-nanogpt .
docker run -it --rm --gpus all -v $(pwd):/app modded-nanogpt python data/fineweb.py
docker run -it --rm --gpus all --env-file .env -v $(pwd):/app modded-nanogpt bash run.sh
```

## NVFP4 Kernels

From [Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation](https://arxiv.org/abs/2601.22813) (Panferov et al., 2026). See [kernels/README.md](kernels/README.md).

## Training Log

See [training_log.md](training_log.md).
