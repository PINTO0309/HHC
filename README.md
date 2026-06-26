# HHC

Head Hat Classification. HHC is a binary classifier for cropped head images. 48x48.

https://github.com/user-attachments/assets/d719d64e-8c5a-454c-9dc8-3c02b776a1a9

## Classes

| class_id | label |
| --- | --- |
| 0 | `no_wearing_hat` |
| 1 | `wearing_hat` |

Default input size is `48x48`.

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|115 KB||0.23 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_p_48x48.onnx)|
|N|176 KB||0.41 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_n_48x48.onnx)|
|T|280 KB||0.52 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_t_48x48.onnx)|
|S|495 KB||0.64 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_s_48x48.onnx)|
|L|6.4 MB|0.9650|1.03 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_l_48x48.onnx)|

## Install

```bash
uv sync --all-extras
source .venv/bin/activate
```

<img width="600" alt="dataset_class_ratio" src="https://github.com/user-attachments/assets/4e2ea89a-9d1a-43a2-b6f4-ba58d63cb31c" />

## Data sample

|1|2|3|4|5|6|
|:-:|:-:|:-:|:-:|:-:|:-:|
|<img width="48" height="48" alt="195332" src="https://github.com/user-attachments/assets/30c9d97b-5fbf-4099-b313-2adaece4819b" />|<img width="48" height="48" alt="195273" src="https://github.com/user-attachments/assets/4709836a-ebc9-493d-8e09-51f0055d0cbe" />|<img width="48" height="48" alt="195238" src="https://github.com/user-attachments/assets/75d42c24-2692-4c3d-9682-3974243815e8" />|<img width="48" height="48" alt="wearing_hat_101005" src="https://github.com/user-attachments/assets/094ae745-fc9b-403c-b674-62a99ae768f0" />|<img width="48" height="48" alt="wearing_hat_100098" src="https://github.com/user-attachments/assets/07bf4469-11bf-48b6-b75e-3f22cfb46138" />|<img width="48" height="48" alt="wearing_hat_100017" src="https://github.com/user-attachments/assets/72fc26f3-a2ca-4233-9709-bbab25c1aad7" />|

## Demo

The demo script needs a YOLO whole-body detector ONNX/TFLite model and an HHC sunglasses classifier ONNX model.
Place the detector model in the repository root, or pass its path with `--model`.
Use the ONNX file exported by training for `--hhc_model`.

```bash
uv run python demo_hhc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--hhc_model hhc_is_l_48x48.onnx \
--images_dir path/to/images \
--execution_provider cpu \
--disable_waitKey
```

For a video file:

```bash
uv run python demo_hhc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--hhc_model hhc_is_l_48x48.onnx \
--video path/to/video.mp4 \
--execution_provider cpu
```

For a camera:

```bash
uv run python demo_hhc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--hhc_model hhc_is_l_48x48.onnx \
--video 0 \
--execution_provider cpu \
--disable_generation_identification_mode \
--disable_gender_identification_mode \
--disable_left_and_right_hand_identification_mode \
--disable_headpose_identification_mode
```
```bash
uv run python demo_hhc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--hhc_model hhc_is_l_48x48.onnx \
--video 0 \
--execution_provider cuda \
--disable_generation_identification_mode \
--disable_gender_identification_mode \
--disable_left_and_right_hand_identification_mode \
--disable_headpose_identification_mode
```

Processed still images are saved under `output/`.
Video input is also recorded to `output.mp4` by default; add `--disable_video_writer` to skip recording.
Use `--execution_provider cuda` or `--execution_provider tensorrt` when the required ONNXRuntime GPU/TensorRT environment is available.

## Train

```bash
SIZE=48x48
VAR=p
uv run python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 1 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=n
uv run python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 2 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=t
uv run python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 3 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=s
uv run python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=l
uv run python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 8 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
