#!/bin/bash

# Set the root directory for storing weights
ROOT_DIR="./weights"
mkdir -p "$ROOT_DIR"

# Define modality-to-model mapping
declare -A MODALITY_MODELS
MODALITY_MODELS=(
    ["infrared"]="stylebooth clip-vit-large-patch14 step-210000"
    ["depth"]="depth_anything_v2"
    ["event"]="none"
    ["normal"]="dsine"
    ["paint"]="paint_transformer"
    ["sketch"]="anime_to_sketch"
)

# Define model download URLs
declare -A MODEL_URLS
MODEL_URLS=(
    ["stylebooth"]="https://huggingface.co/scepter-studio/stylebooth/resolve/main/models/stylebooth-tb-5000-0.bin?download=true"
    ["clip-vit-large-patch14"]="https://huggingface.co/openai/clip-vit-large-patch14"
#    ["step-210000"]="https://drive.google.com/drive/folders/1bXe9MGJN_qvBnwONZ9uVImSJj-visH0m?usp=sharing"
    ["step-210000"]="lsxi77777/MINIMA@MINIMA_engine_models:step-210000"
    ["depth_anything_v2"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
    ["dsine"]="https://drive.google.com/uc?id=1Wyiei4a-lVM6izjTNoBLIC5-Rcy4jnaC"
    ["paint_transformer"]="https://drive.google.com/uc?id=1NDD54BLligyr8tzo8QGI5eihZisXK1nq"
    ["anime_to_sketch"]="https://drive.google.com/uc?id=1cf90_fPW-elGOKu5mTXT5N1dum-XY_46"
)

# Define paths for storing weights
declare -A MODEL_PATHS
MODEL_PATHS=(
    ["stylebooth"]="$ROOT_DIR/stylebooth/stylebooth-tb-5000-0.bin"
    ["clip-vit-large-patch14"]="$ROOT_DIR/clip-vit-large-patch14/"
    ["step-210000"]="$ROOT_DIR/stylebooth/"
    ["depth_anything_v2"]="$ROOT_DIR/depth_anything_v2/depth_anything_v2_vitl.pth"
    ["dsine"]="$ROOT_DIR/dsine/dsine.pt"
    ["paint_transformer"]="$ROOT_DIR/paint_transformer/model.pth"
    ["anime_to_sketch"]="$ROOT_DIR/anime_to_sketch/improved.bin"
)

# Function to download a model
download_model() {
    local model=$1
    local TARGET_PATH="${MODEL_PATHS[$model]}"
    mkdir -p "$(dirname "$TARGET_PATH")"
    echo "Downloading $model to $TARGET_PATH..."

    if [[ ${MODEL_URLS[$model]} == *"/folders/"* ]]; then
        # If it's a Google Drive folder, use gdown --folder
        FOLDER_ID=$(echo "${MODEL_URLS[$model]}" | grep -oP 'folders/\K[^?]+')
        gdown --folder "$FOLDER_ID" -O "$TARGET_PATH"
#        echo "Google Drive folder download not supported."
    elif [[ ${MODEL_URLS[$model]} == *"drive.google.com"* ]]; then
        # If it's a Google Drive file, use gdown
        gdown "${MODEL_URLS[$model]}" -O "$TARGET_PATH"
#        echo "Google Drive download not supported."
    elif [[ ${MODEL_URLS[$model]} == *"huggingface.co"* ]]; then
        if [[ ${MODEL_URLS[$model]} == *"/resolve/main/"* ]]; then
            # Download a specific file from Hugging Face
            wget -O "$TARGET_PATH" "${MODEL_URLS[$model]}"
#            echo "Hugging Face download not supported."
        else
            # Download the full Hugging Face repo
            REPO_NAME=$(echo "${MODEL_URLS[$model]}" | grep -oP 'huggingface.co/\K[^/]*/[^/]*')
            huggingface-cli download "$REPO_NAME" --local-dir "$TARGET_PATH" --local-dir-use-symlinks False
#            echo "Hugging Face download not supported."
        fi
    elif [[ ${MODEL_URLS[$model]} == *"@"* ]]; then
        REPO_PATH=$(echo "${MODEL_URLS[$model]}" | cut -d'@' -f1)
        BRANCH_AND_SUBDIR=$(echo "${MODEL_URLS[$model]}" | cut -d'@' -f2)
        BRANCH=$(echo "$BRANCH_AND_SUBDIR" | cut -d':' -f1)
        SUBDIR=$(echo "$BRANCH_AND_SUBDIR" | cut -d':' -f2)

        huggingface-cli download $REPO_PATH \
            --revision $BRANCH \
            --include "$SUBDIR/*" \
            --local-dir "$TARGET_PATH" \
            --local-dir-use-symlinks False
    else
        # Otherwise, use wget
        wget -O "$TARGET_PATH" "${MODEL_URLS[$model]}"
#        echo "Download not supported."

    fi
}


while true; do
    echo "Choose download mode:"
    select mode in "By Modality" "By Model" "Exit"; do
        case $mode in
            "By Modality")
                while true; do
                    echo "Select a modality:"
                    select modality in "${!MODALITY_MODELS[@]}" "Back"; do
                        if [[ $modality == "Back" ]]; then
                            break
                        elif [[ $modality == "event" ]]; then
                            echo "No download required for Event Generation."
                            break
                        elif [[ -n ${MODALITY_MODELS[$modality]} ]]; then
                            echo "Downloading models for $modality..."
                            for model in ${MODALITY_MODELS[$modality]}; do
                                download_model "$model"
                            done
                            break
                        else
                            echo "Invalid option, please try again."
                        fi
                    done
                    break
                done
                break
                ;;
            "By Model")
                while true; do
                    echo "Select a model to download:"
                    select model in "${!MODEL_URLS[@]}" "Download all" "Back"; do
                        case $model in
                            "Download all")
                                for key in "${!MODEL_URLS[@]}"; do
                                    download_model "$key"
                                done
                                break
                                ;;
                            "Back")
                                break
                                ;;
                            *)
                                if [[ -n ${MODEL_URLS[$model]} ]]; then
                                    download_model "$model"
                                else
                                    echo "Invalid option, please try again."
                                fi
                                ;;
                        esac
                    done
                    break
                done
                break
                ;;
            "Exit")
                echo "Exiting."
                exit 0
                ;;
            *)
                echo "Invalid option, please try again."
                ;;
        esac
    done
done