# FoundationStereo: Zero-Shot Stereo Matching
# Installation

We've tested on Linux with GPU 3090, 4090, A100, V100, Jetson Orin. Other GPUs should also work, but make sure you have enough memory

```
conda env create -f environment.yml
conda run -n foundation_stereo pip install flash-attn
conda activate foundation_stereo
```

Note that `flash-attn` needs to be installed separately to avoid [errors during environment creation](https://github.com/NVlabs/FoundationStereo/issues/20).


# Model Weights
- Download the foundation model for zero-shot inference on your data. Put the entire folder (e.g. `23-51-11`) under `./pretrained_models/`.


| Model     | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| [23-51-11](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)  | Our best performing model for general use, based on Vit-large               |
| [11-33-40](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)  | Slightly lower accuracy but faster inference, based on Vit-small            |
| [NVIDIA-TAO](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo)       | For commercial usage (adapted from Vit-small model)                 |

# Run demo
```
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/
```
# Run stereo to depth
```
python scripts/run_runzenew.py --left_dir=left_rectified --right_dir=right_lectified --intrinsic_file=K.txt --ckpt_dir=pretrained_models/23-51-11/model_best_bp2.pth --out_dir=output, type=str
```


