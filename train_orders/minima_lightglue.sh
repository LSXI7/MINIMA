#!/bin/bash -l

ORDER_PATH=$(dirname $(readlink -f "$0"))
MINIMA_PATH="${ORDER_PATH}/../"
LIGHTGLUE_DIR="${MINIMA_PATH}/third_party/glue_factory_minima/"
CONFIG_PATH="${ORDER_PATH}/minima_lightglue_train_config.yaml"
SAVE_DIR="${MINIMA_PATH}/train_results/minima_lightglue/"


export PYTHONPATH=$LIGHTGLUE_DIR:$PYTHONPATH
cd $LIGHTGLUE_DIR


# Data
readarray -t modality_list < <(yq '.MINIMA_LIGHTGLUE.dataset.modality[]' "$CONFIG_PATH")
modality_args=(--modality_list "${modality_list[@]}")
experiment=$(yq -r '.MINIMA_LIGHTGLUE.train.experiment' "$CONFIG_PATH")
lr_scale=$(yq '.MINIMA_LIGHTGLUE.train.lr_scale' "$CONFIG_PATH")
load_features=$(yq '.MINIMA_LIGHTGLUE.dataset.load_features' "$CONFIG_PATH")

# Weights
pretrained=$(yq '.MINIMA_LIGHTGLUE.weight.pre_trained' "$CONFIG_PATH")
pretrained_path=$(yq -r '.MINIMA_LIGHTGLUE.weight.pre_trained_path' "$CONFIG_PATH")
pre_trained_path="${MINIMA_PATH}/${pretrained_path}"
resume=$(yq '.MINIMA_LIGHTGLUE.weight.resume_from_checkpoint' "$CONFIG_PATH")

if [[ "$pretrained" == "true" && "$resume" == "true" ]]; then
    echo "[Error] Cannot enable both 'pre_trained' and 'resume_from_checkpoint'." >&2
    exit 1
fi

extra_args=""

if [[ "$pretrained" == "true" && -n "$pretrained_path" && "$pretrained_path" != "null" ]]; then
    extra_args=" --ckpt_path $pre_trained_path"
elif [[ "$resume" == "true" ]]; then
    extra_args=" --restore"
fi

python -m gluefactory.train sp+lg_megadepth_depth_lower_lr \
    --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml \
    --distributed \
    --no_eval_0  \
    --lr_scale=${lr_scale} \
    --experiment=${experiment} \
    --save_dir=${SAVE_DIR} \
    "${modality_args[@]}" \
    $extra_args \
    data.load_features.do=${load_features} \
