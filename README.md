# HHC

Head Hat Classification. HHC is a binary classifier for cropped head images.

## Classes

| class_id | label |
| --- | --- |
| 0 | `no_wearing_hat` |
| 1 | `wearing_hat` |

Default input size is `32x32`.

## Dataset

Expected image layout:

```text
data/
  train/
    no_wearing_hat/<source_id>/*.png
    wearing_hat/<source_id>/*.png
  val/
    no_wearing_hat/<source_id>/*.png
    wearing_hat/<source_id>/*.png
```

Current dataset counts:

| split | no_wearing_hat | wearing_hat |
| --- | ---: | ---: |
| train | 8,383 | 8,383 |
| val | 932 | 932 |

## Commands

Build the parquet dataset:

```bash
python 01_build_hat_parquet.py --root data --output data/dataset.parquet
```

Plot the class ratio:

```bash
python 02_plot_dataset_pie.py --input data/dataset.parquet
```

Train:

```bash
hhc train --data_root data --output_dir runs/hhc_32x32 --image_size 32
```

Predict with a checkpoint:

```bash
hhc predict --checkpoint runs/hhc_32x32/hhc_best_epoch0001_f1_0.9000.pt --inputs path/to/images
```

Export ONNX:

```bash
hhc exportonnx --checkpoint runs/hhc_32x32/hhc_best_epoch0001_f1_0.9000.pt --output hhc_32x32.onnx
```

Run the whole-body demo with the HHC ONNX classifier:

```bash
python demo_hhc.py --model yolomit_t_wholebody28_1x3x480x640.onnx --hhc_model hhc_32x32.onnx --images_dir path/to/images
```

## Train

```bash
SIZE=32x32
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
SIZE=32x32
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
SIZE=32x32
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
SIZE=32x32
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
SIZE=32x32
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
