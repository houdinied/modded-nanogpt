#!/bin/bash
SAVE_DIR="fineweb10B"
TRAIN_BASE_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_train_"
MAX_SHARDS=103

download() {
    local FILE_URL=$1
    local FILE_NAME=$(basename $FILE_URL | cut -d'?' -f1)
    local FILE_PATH="${SAVE_DIR}/${FILE_NAME}"
    if [ -f "$FILE_PATH" ]; then
        echo "Skipping $FILE_NAME (already exists)"
        return
    fi
    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    echo "Downloaded $FILE_NAME"
}
export -f download
export SAVE_DIR

job_count=0
for i in $(seq -f "%06g" 1 $MAX_SHARDS); do
    FILE_URL="${TRAIN_BASE_URL}${i}.bin?download=true"
    download "$FILE_URL" &
    ((job_count++))
    if (( job_count >= 40 )); then
        wait -n
        ((job_count--))
    fi
done
wait
echo "Done! Downloaded $MAX_SHARDS training shards to $SAVE_DIR"
