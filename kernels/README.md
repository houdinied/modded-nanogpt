# NVFP4 Kernels

From [Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation](https://arxiv.org/abs/2601.22813) by Panferov et al., 2026.

Kernels tuned for B200 (sm100). Requires CUDA 13.0, PyTorch 2.10+, Python 3.11+.

## Install

```bash
cd kernels
pip install --no-build-isolation .
```

## Citation

```bibtex
@misc{panferov2026quartetiiaccuratellm,
      title={Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation}, 
      author={Andrei Panferov and Erik Schultheis and Soroush Tabesh and Dan Alistarh},
      year={2026},
      eprint={2601.22813},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.22813}, 
}
```
