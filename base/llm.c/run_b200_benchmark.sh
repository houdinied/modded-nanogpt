#!/bin/bash
# GPT-2 124M benchmark on 8x B200
# Measures wall time, tokens/sec, and targets val_loss <= 3.28

set -e

out_dir="log_gpt2_124M_b200_bench"
rm -rf "$out_dir"
mkdir -p "$out_dir"

echo "=== GPT-2 124M Benchmark on 8x B200 ==="
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
START_EPOCH=$(date +%s%N)

# Same hyperparams as run_gpt2_124M.sh
# -v 250: eval every 250 steps
# -s 20000: max steps (will stop via done file at 18865)
# -g 144: gelu recompute
# -h 1: hellaswag eval
# -b 64: micro batch size
# -t 1024: sequence length
# -d 524288: total batch size in tokens
# -r 0: resume off (fresh run)
# -z 1: zero stage 1
# -c 0.1: weight decay
# -l 0.0006: learning rate
# -q 0.0: no dropout
# -u 700: warmup steps
# -n 5000: checkpoint every 5000 steps
# -y 0: no resume
# -e d12: GPT-2 124M architecture

mpirun -np 8 \
    --allow-run-as-root \
    ./train_gpt2cu \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o "$out_dir" \
    -v 250 -s 18865 -g 144 \
    -h 1 \
    -b 64 -t 1024 \
    -d 524288 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -n 5000 \
    -y 0 \
    -e "d12"

END_EPOCH=$(date +%s%N)
ELAPSED_MS=$(( (END_EPOCH - START_EPOCH) / 1000000 ))
ELAPSED_S=$(echo "scale=2; $ELAPSED_MS / 1000" | bc)
ELAPSED_M=$(echo "scale=2; $ELAPSED_MS / 60000" | bc)

echo ""
echo "=== Benchmark Complete ==="
echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Wall time: ${ELAPSED_S}s (${ELAPSED_M} min)"
echo "Log: $out_dir/main.log"
