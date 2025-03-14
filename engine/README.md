# Data Engine of MINIMA (CVPR 2025) for Multi-modal Image Generation

MINIMA Data Engine is an open-source project that integrates cutting-edge methods from the community to enable
multi-modal transformations.  
Currently, the engine supports the following modalities:
<div align="center">

<img src="figs/example.png" width="800" alt="framework">

</div>

## Online Demo

The online demo is under development. Stay tuned!

## Environment Setup

The engine requires the following dependencies:

```bash
pip install -r engine_extra_requirements.txt
```

## Checkpoints Preparation

We recommend placing all checkpoints in the `engine/weights` directory.  
The `download_weights.sh` script can be used to download all the required weights and place them in the correct
directory following the instructions below:

```bash
cd engine
bash download_weights.sh
```

You can also download the weights manually and place them in the `engine/weights` directory.

<p></p> <details> <summary><b> Weights Structure </b></summary>  

The directory structure should be like this:

```
weights/
├── stylebooth/
│ ├── step-210000/
│ └── stylebooth-tb-5000-0.bin
├── clip-vit-large-patch14/
│ ├── tokenizer.json
│ └── ...
├── depth_anything_v2/
│ └── depth_anything_v2_vitl.pth
├── dsine/
│ └── dsine.pt
├── paint_transformer/
│ └── model.pth
└── anime_to_sketch/
  └── improved.bin
```

</details>
<p></p>


<p></p> <details> <summary><b> Infrared Generation </b></summary>  

The infrared generation code is based on [scepter](https://github.com/modelscope/scepter)

Please
download the weights
from [styleBooth weights](https://huggingface.co/scepter-studio/stylebooth/tree/main/models), [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14).
And our style tuner is available for download
at [step-210000](https://drive.google.com/drive/folders/1bXe9MGJN_qvBnwONZ9uVImSJj-visH0m?usp=sharing)

> **NOTE:** Generation a 1024x1024 image requires a GPU with about 12GB of memory.
</details>
<p></p>

<p></p> <details> <summary><b> Depth Generation </b></summary>  

The depth generation code is based on [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

Please
download the weights
from [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)

</details>
<p></p>


<p></p><details>
  <summary><b> Event Generation </b></summary>

The event generation module is a simple simulation implemented with basic code.

> **NOTE:** Since this is a simulated process, no checkpoint is required.

</details><p></p>


<p></p> <details> <summary><b> Normal Generation </b></summary>  

The normal generation code is based on [DSINE](https://github.com/baegwangbin/DSINE)

Please download [weights](https://drive.google.com/file/d/1Wyiei4a-lVM6izjTNoBLIC5-Rcy4jnaC/view?usp=sharing)

</details>
<p></p>

<p></p> <details> <summary><b> Sketch Generation </b></summary>  

The sketch generation code is based on [Anime2Sketch](https://github.com/Mukosame/Anime2Sketch)

Please download
the [weights](https://drive.google.com/file/d/1cf90_fPW-elGOKu5mTXT5N1dum-XY_46/view)

</details>
<p></p>

<p></p> <details> <summary><b> Paint Generation </b></summary>  

The paint generation code is based on [PaintTransformer](https://github.com/Huage001/PaintTransformer)

Please download [weights](https://drive.google.com/file/d/1NDD54BLligyr8tzo8QGI5eihZisXK1nq/view)

</details>
<p></p>

## MINIMA Data Engine

To run the engine, you can use the following command:

```bash
cd engine
python modality_engine.py --modality <modality> --input_path <input_path> --output_dir <output_dir>
# --modality: Choose from [infrared, depth, event, normal, sketch, paint]
# --input_path: Supports both a single image or a directory that contains images

# Example
python modality_engine.py --modality infrared --input_path ./figs/origin_image.jpg --output_dir './result'
```

## Acknowledgements

We sincerely appreciate the contributions from the open-source community.

Special thanks to:

- [Scepter](https://github.com/modelscope/scepter)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [DSINE](https://github.com/baegwangbin/DSINE)
- [Anime2Sketch](https://github.com/Mukosame/Anime2Sketch)
- [PaintTransformer](https://github.com/Huage001/PaintTransformer)

Your support and feedback help improve MINIMA Data Engine. We welcome contributions and collaborations from
the community!