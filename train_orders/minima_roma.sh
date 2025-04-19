#!/bin/bash -l

ORDER_PATH=$(dirname $(readlink -f "$0"))
MINIMA_PATH="${ORDER_PATH}/../"
ROMA_DIR="${MINIMA_PATH}/third_party/RoMa_minima/"
CONFIG_PATH="${ORDER_PATH}/minima_roma_train_config.yaml"
SAVE_DIR="${MINIMA_PATH}/train_results/minima_roma/"

export PYTHONPATH=$ROMA_DIR:$PYTHONPATH
cd $ROMA_DIR

# Data
readarray -t modality_list < <(yq '.MINIMA_ROMA.dataset.modality[]' "$CONFIG_PATH")
modality_args=(--modality_list "${modality_list[@]}")

# DDP
n_nodes=$( yq '.MINIMA_ROMA.train.n_nodes' "$CONFIG_PATH" )
n_gpus_per_node=$( yq '.MINIMA_ROMA.train.n_gpus_per_node' "$CONFIG_PATH" )
batch_size=$(yq '.MINIMA_ROMA.train.batch_size' "$CONFIG_PATH")
lr_scale=$(yq '.MINIMA_ROMA.train.lr_scale' "$CONFIG_PATH")
exp_name=$(yq -r '.MINIMA_ROMA.train.exp_name' "$CONFIG_PATH")

# Weights
pretrained=$(yq '.MINIMA_ROMA.weight.pre_trained' "$CONFIG_PATH")
pretrained_path=$(yq -r '.MINIMA_ROMA.weight.pre_trained_path' "$CONFIG_PATH")
pre_trained_path="${MINIMA_PATH}/${pretrained_path}"
resume=$(yq '.MINIMA_ROMA.weight.resume_from_checkpoint' "$CONFIG_PATH")

if [[ "$pretrained" == "true" && "$resume" == "true" ]]; then
    echo "[Error] Cannot enable both 'pre_trained' and 'resume_from_checkpoint'." >&2
    exit 1
fi

extra_args=""

if [[ "$pretrained" == "true" && -n "$pretrained_path" && "$pretrained_path" != "null" ]]; then
    extra_args=" --ckpt_path $pre_trained_path"
elif [[ "$resume" == "true" ]]; then
    extra_args=" --resume"
fi


torchrun --nproc_per_node=${n_gpus_per_node} --nnodes=${n_nodes} \
    ${ROMA_DIR}/train_roma_outdoor.py \
    --e_name ${exp_name} \
    --lr_scale=${lr_scale} \
    "${modality_args[@]}" \
    $extra_args \
    --save_dir=${SAVE_DIR} \
    --gpu_batch_size=${batch_size} \
    --dont_log_wandb \