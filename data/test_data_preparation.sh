#!/bin/bash

# Step 1: Change to the 'data' directory
cd data || { echo "Directory 'data' not found. Please create it and retry."; exit 1; }

# Step 2: Download the dataset using openxlab
openxlab dataset download --dataset-repo lsxi7/MINIMA --source-path /Megadepth-1500-syn

# Step 3: Verify the downloaded folder exists
if [ ! -d "lsxi7___MINIMA/Megadepth-1500-syn" ]; then
    echo "Download failed or the folder 'lsxi7___MINIMA/Megadepth-1500-syn' does not exist."
    exit 1
fi

# Step 4: Define the source and target directories
root_dir=$(pwd)
echo  $root_dir
SOURCE_DIR="$root_dir/lsxi7___MINIMA/Megadepth-1500-syn"
TARGET_DIR="megadepth/train"

# Step 5: Ensure the target directory exists
mkdir -p "$TARGET_DIR"

# Step 6: Iterate through modality folders and extract tar.gz files
for MODALITY in Depth Event Infrared Normal Sketch Paint; do
    MODALITY_DIR="$SOURCE_DIR/$MODALITY"
    MODALITY_LOWER=$(echo "$MODALITY" | tr '[:upper:]' '[:lower:]')
    if [ -d "$MODALITY_DIR" ]; then
        TAR_FILE="$MODALITY_DIR/${MODALITY_LOWER}_Undistorted_SfM.tar.gz"
        if [ -f "$TAR_FILE" ]; then
            # Extract the tar.gz file
            tar -xzf "$TAR_FILE" -C "$MODALITY_DIR"
            EXTRACTED_DIR="$MODALITY_DIR/Undistorted_SfM"
            echo $EXTRACTED_DIR
            echo $TARGET_DIR/$MODALITY_LOWER
            # Check if extraction was successful
            if [ -d "$EXTRACTED_DIR" ]; then
                # Create a symbolic link ixn the target directory
                ln -s "$EXTRACTED_DIR" "$TARGET_DIR/$MODALITY_LOWER"
                echo "Linked $MODALITY to $TARGET_DIR/$MODALITY_LOWER"
            else
                echo "Extraction failed for $MODALITY. Directory $EXTRACTED_DIR not found."
            fi
        else
            echo "Tar file not found for $MODALITY: $TAR_FILE"
        fi
    else
        echo "Modality directory not found: $MODALITY_DIR"
    fi
done

echo "Data preparation completed."
