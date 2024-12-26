#!/bin/bash

# Ensure the script is run from the "weights" folder
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$PWD" != "$SCRIPT_DIR" ]]; then
    echo "Please run this script from the 'weights' directory."
    exit 1
fi

# URLs of the weight files
MINIMA_LG="https://github.com/LSXI7/storage/releases/download/MINIMA/minima_lightglue.pth"
MINIMA_LoFTR='https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt'
MINIMA_RoMa='https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth'

# Download the files
FILES=(
    "$MINIMA_LG"
    "$MINIMA_LoFTR"
    "$MINIMA_RoMa"
)

for FILE_URL in "${FILES[@]}"; do
    FILE_NAME=$(basename "$FILE_URL")
    if [[ -f "$FILE_NAME" ]]; then
        echo "$FILE_NAME already exists, skipping download."
    else
        echo "Downloading $FILE_NAME..."
        curl -L -O "$FILE_URL"
        if [[ $? -eq 0 ]]; then
            echo "$FILE_NAME downloaded successfully."
        else
            echo "Failed to download $FILE_NAME."
        fi
    fi
done
