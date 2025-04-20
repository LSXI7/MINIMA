# MINIMA Training Instruction

This repository provides training scripts for three versions of MINIMA:

- `minima_lightglue`
- `minima_loftr`
- `minima_roma`

> **Note:** Each version is based on its original implementation, and currently they are maintained independently.
> Therefore, they require **different environments and index formats** for training. We provide ready-to-use training
> scripts for each variant under the `train_orders/` directory.

To start training, simply run the corresponding script from the root of the MINIMA project.

please ensure that you have the required dependencies installed in your environment.

```bash
git submodule update --init --recursive
git submodule update --recursive --remote
```

By default, all pretrained weights are expected to be stored in the `MINIMA/weights/` directory.
All results will be saved in the `MINIMA/train_results/` directory.

---

## Dataset Setup

We train our models using the **undistorted images** from the
original [MegaDepth_v1](https://www.cs.cornell.edu/projects/megadepth/) dataset. On top of that, we construct **six
additional aligned modalities** using a data engine, enabling training with multi-modal pairs.

You can download the additional modalities from either of the following platforms:

- [OpenXLab](https://openxlab.org.cn/datasets/lsxi7/MINIMA)
- [Hugging Face](https://huggingface.co/datasets/lsxi77777/MegaDepth-Syn)

> The MegaDepth dataset (~200GB) must be downloaded manually from the original website. It includes matched image pairs
> and depth maps. You will also need additional **index files and labels** for training, which are provided separately
> for each method below.

The original MegaDepth dataset includes `.JPG` and `.png` files.
To ensure consistency, all images were renamed or converted to `.jpg` format using the script below:

```bash
python train_orders/megadepth_rename.py
```

### Data Structure

We recommend organizing both the original **MegaDepth** dataset and the multi-modal **MegaDepth-Syn** under
the `MINIMA/data/` directory.    
Below is the recommended folder structure:

```
MINIMA/data/megadepth/
├── train/
│ ├── phoenix/              # MegaDepth
│ ├── infrared/             # MegaDepth-Syn
│ │ ├── phoenix/
│ │ └── Undistorted_SfM/    # for test and val
│ ├── [modality]
│ └── ...
└── test/Undistorted_SfM/   # MegaDepth-1500
```

To make the dataset available to each training script, create symbolic links from your dataset directory to the
expected `third_party` locations:

```bash
# LightGlue
ln -s  /path/to/MINIMA/data/megadepth/train/* ./third_party/glue_factory_minima/data/megadepth/

# LoFTR
ln -s  /path/to/MINIMA/data/megadepth/train/ ./third_party/LoFTR_minima/data/megadepth/

# RoMA
ln -s  /path/to/MINIMA/data/megadepth/train/* ./third_party/RoMa_minima/data/megadepth/
```

---

## MINIMA LightGlue

### Index

Download the index files from [here](https://cvg-data.inf.ethz.ch/megadepth/scene_info.tar.gz):

```bash
tar xf scene_info.tar.gz -C ./third_party/glue_factory_minima/data/megadepth/
python train_orders/lightglue_index_preparation.py
```

### Training

```bash
bash train_orders/minima_lightglue.sh
# default load_feature is False
```

you can cache the local features before training to speed up the training process.

```bash
cd third_party/glue_factory_minima
python -m gluefactory.scripts.export_megadepth --method sp --num_workers 8
```

Set the `load_feature` to `True` in the `train_orders/minima_lightglue_train_config.yaml` file and run the training
script:

```bash
bash train_orders/minima_lightglue.sh
```

Training parameters and modal options can be customized in the
accompanying `train_orders/minima_lightglue_train_config.yaml` file.

---

## MINIMA LoFTR

### Index

Download the LoFTR index files
from [Google Drive](https://drive.google.com/file/d/1YMAAqCQLmwMLqAkuRIJLDZ4dlsQiOiNA/view?usp=drive_link).

```bash
mkdir -p tmp_unpack
tar xf megadepth_indices.tar -C tmp_unpack
mv tmp_unpack/megadepth_indices/* ./third_party/LoFTR_minima/data/megadepth/
rm -rf tmp_unpack
python train_orders/loftr_index_preparation.py
```

### Training

```bash
bash train_orders/minima_loftr.sh
```

Training parameters and modal options can be customized in the
accompanying `train_orders/minima_loftr_train_config.yaml` file.

---

## MINIMA RoMA

### Index

Download the RoMA index files from
this [release](https://github.com/Parskatt/storage/releases/download/prep_scene_info/prep_scene_info.tar).

```bash
tar xf prep_scene_info.tar -C ./third_party/roma_minima/data/megadepth/
python train_order/roma_index_preparation.py
```

### Training

```bash
bash train_orders/minima_roma.sh
```

Training parameters and modal options can be customized in the
accompanying `train_orders/minima_roma_train_config.yaml` file.

---

## Acknowledgement

We would like to thank the authors of the original implementations
of [glue-factory](https://github.com/cvg/glue-factory), [LoFTR](https://github.com/zju3dv/LoFTR),
and [RoMA](https://github.com/Parskatt/RoMa) for making their
excellent work available to the community. MINIMA builds upon their efforts with additional extensions for multi-modal
training.

