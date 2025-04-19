#!/bin/bash -l

ORDER_PATH=$(dirname $(readlink -f "$0"))
MINIMA_PATH="${ORDER_PATH}/../"
LOFTR_DIR="${MINIMA_PATH}/third_party/LoFTR_minima/"
CONFIG_PATH="${ORDER_PATH}/minima_loftr_train_config.yaml"
SAVE_DIR="${MINIMA_PATH}/train_results/minima_loftr/"

# conda activate loftr
export PYTHONPATH=$LOFTR_DIR:$PYTHONPATH
cd $LOFTR_DIR


# Data
TRAIN_IMG_SIZE=$( yq '.MINIMA_LOFTR.train.image_size' "$CONFIG_PATH" )
# to reproduced the results in our paper, please use:
# TRAIN_IMG_SIZE=840
data_cfg_path="${LOFTR_DIR}/configs/data/megadepth_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="${LOFTR_DIR}/configs/loftr/outdoor/loftr_ds_dense.py"
readarray -t modality_list < <(yq '.MINIMA_LOFTR.dataset.modality[]' "$CONFIG_PATH")
modality_args=(--modality_list "${modality_list[@]}")

# DDP
n_nodes=$( yq '.MINIMA_LOFTR.train.n_nodes' "$CONFIG_PATH" )
n_gpus_per_node=$( yq '.MINIMA_LOFTR.train.n_gpus_per_node' "$CONFIG_PATH" )
torch_num_workers=$(yq '.MINIMA_LOFTR.train.torch_num_workers' "$CONFIG_PATH")
batch_size=$(yq '.MINIMA_LOFTR.train.batch_size' "$CONFIG_PATH")
pin_memory=$(yq '.MINIMA_LOFTR.train.pin_memory' "$CONFIG_PATH")
exp_name="outdoor-ds-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"
max_epochs=$(yq '.MINIMA_LOFTR.train.max_epochs' "$CONFIG_PATH")
save_top_k=$(yq '.MINIMA_LOFTR.train.save_top_k' "$CONFIG_PATH")
lr_scale=$(yq '.MINIMA_LOFTR.train.lr_scale' "$CONFIG_PATH")

# Weights
pretrained=$(yq '.MINIMA_LOFTR.weight.pre_trained' "$CONFIG_PATH")
pretrained_path=$(yq -r '.MINIMA_LOFTR.weight.pre_trained_path' "$CONFIG_PATH")
pre_trained_path="${MINIMA_PATH}/${pretrained_path}"
resume=$(yq '.MINIMA_LOFTR.weight.resume_from_checkpoint' "$CONFIG_PATH")
resume_path=$(yq '.MINIMA_LOFTR.weight.resume_path' "$CONFIG_PATH")

if [[ "$pretrained" == "true" && "$resume" == "true" ]]; then
    echo "[Error] Cannot enable both 'pre_trained' and 'resume_from_checkpoint'." >&2
    exit 1
fi

extra_args=""

if [[ "$pretrained" == "true" && -n "$pretrained_path" && "$pretrained_path" != "null" ]]; then
    extra_args=" --ckpt_path $pre_trained_path"
elif [[ "$resume" == "true" && -n "$resume_path" && "$resume_path" != "null" ]]; then
    extra_args=" --resume_from_checkpoint $resume_path"
fi

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=1 \
    --flush_logs_every_n_steps=1 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=${max_epochs} \
    --save_top_k=${save_top_k} \
    --lr_scale=${lr_scale} \
    "${modality_args[@]}" \
    --save_dir=${SAVE_DIR} \
    $extra_args
