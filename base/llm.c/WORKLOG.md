## Getting llm.c to compile on CUDA 13.0 + B200

### Problem 1: `cudaMemAdvise` API change

CUDA 13.0 changed `cudaMemAdvise` — the 4th arg is now a `cudaMemLocation` struct, not an int.

`llmc/cuda_utils.cuh:220` had:
```c
cudaCheck_(cudaMemAdvise(*out, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId), file, line);
```

Fixed to:
```c
cudaMemLocation loc = {};
loc.type = cudaMemLocationTypeHostNuma;
loc.id = 0;
cudaCheck_(cudaMemAdvise(*out, bytes, cudaMemAdviseSetPreferredLocation, loc), file, line);
```

### Problem 2: cuDNN frontend missing

`make train_gpt2cu USE_CUDNN=1` needs the cudnn-frontend headers. They're not installed by the cudnn apt packages.

```bash
cd ~ && git clone https://github.com/NVIDIA/cudnn-frontend.git
```

Makefile looks for `~/cudnn-frontend/include` automatically.

### Problem 3: `-lnvrtc` missing from linker flags

New cudnn-frontend uses NVRTC internally. The Makefile didn't link it. Got a wall of `undefined reference to nvrtcCreateProgram` etc.

In `Makefile`, changed:
```
NVCC_LDFLAGS += -lcudnn
```
to:
```
NVCC_LDFLAGS += -lcudnn -lnvrtc
```

### Problem 4: B200 not in MFU GPU database

`llmc/mfu.h` didn't have a Blackwell entry, so MFU showed -100.0%.

Added Blackwell PerfData and a `"NVIDIA B200"` entry to `gpu_db[]`.

### Problem 5: mpirun as root

Need both env vars exported:
```bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

### Problem 6: tinyshakespeare too small for 8 GPUs

Default data (305K tokens) fails the assertion `shard_ntok >= num_processes * B * T + 1` with 8 GPUs. Need fineweb10B data (~20GB). Downloaded pre-tokenized shards:
```bash
cd dev/data && bash fineweb.sh 103
```
Or manually curl 103 shards into `dev/data/fineweb10B/`.
