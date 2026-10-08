# Optical-SAR Fusion for Cloud Removal via Cross-Modal Mutual Knowledge Distillation

Reference implementation of the proposed method on **SEN12MS-CR**: training and evaluation.

## 1. Environment

Python 3.9.19, PyTorch 2.4.1 (CUDA 12.1); results were produced on a single RTX 4090.

```bash
conda create -n mckd python=3.9 -y
conda activate mckd
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Layout

```
main/              train.py, test.py, dataloader.py, loss.py, model/mckd.py, utils/
splits/            splits.csv
dataset_prep/      split.txt, flatten_scenes.py
checkpoints/       the two .pkl files (see section 6)
SEN12MS_dataset/   you create this (see section 3)
```

Run everything from inside `main/` — the data paths in the code are relative.

## 3. Data

Download **SEN12MS-CR** from <https://mediatum.ub.tum.de/1554803> and lay it out as

```
SEN12MS_dataset/
  ROIs*_<season>_s1/         s1_N/        ..._s1_N_p0.tif
  ROIs*_<season>_s2/         s2_N/        ..._s2_N_p0.tif
  ROIs*_<season>_s2_cloudy/  s2_cloudy_N/ ..._s2_cloudy_N_p0.tif
```

In each 256×256 patch, `s2` is the cloud-free 13-band target, `s2_cloudy` the cloudy input
and `s1` the 2-band (VV, VH) SAR input. `dataset_prep/flatten_scenes.py` flattens the
downloaded tree and `dataset_prep/split.txt` is the scene-level division.

`splits/splits.csv` is the split behind every reported number — 107143 train / 7176 val /
**7899 test**, selected by the `split` column (`1`/`2`/`3`).

## 4. Train

```bash
cd main

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type My --add_message My_end_KD_teacher --dataset_name Sen12 \
    --input_data_folder ../SEN12MS_dataset \
    --data_list_filepath ../splits/splits.csv \
    --batch_size 12 --lr 1e-4 --optimizer adamw \
    --total_epoch 15 --load_size 256 --num_workers 6
```

One invocation runs both stages: stage 1 trains the teacher, stage 2 distils the student.
About 37 h on one 4090.

To run stage 2 alone against the released teacher, add

```bash
    --student_only True \
    --teacher_path ./result/all_My_My_end_KD_teacher/last.pkl
```

## 5. Test

```bash
cd main

CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_type My --dataset_name Sen12 \
    --weight_path ./result/all_My_My_end_KD_student/last.pkl \
    --input_data_folder ../SEN12MS_dataset \
    --data_list_filepath ../splits/splits.csv \
    --batch_size 1 --num_workers 6
```

Metrics are PSNR, SSIM, MAE and SAM on the full 13-band reconstruction, with no cloud mask
applied. **Keep `--batch_size 1`**: `utils/metric.py` pools PSNR and SAM over the whole
batch, so pooled values move with the batch size.

Expected on the 7899 test patches: **PSNR 30.4002, SSIM 0.9019, MAE 0.0240, SAM 7.7339**.

## 6. Checkpoints

| file | size | architecture |
|---|---:|---|
| `teacher.pkl` | 96.5 MB | `model.mckd.Net(13, 2)` |
| `student.pkl` | 97.8 MB | `model.mckd.Net(13, 2, iss=True)` |

Place `teacher.pkl` at `main/result/all_My_My_end_KD_teacher/last.pkl` and `student.pkl`
at `main/result/all_My_My_end_KD_student/last.pkl`; the scripts read `last.pkl`, not the
asset name. Both are epoch 14 of their 15-epoch run.

Both were saved from `nn.DataParallel`, so every key carries a `module.` prefix. The shipped
scripts wrap the network the same way, which is why it resolves there; into an unwrapped
network, strip exactly one leading `module.` — not with `str.replace`, which would also
mangle the network's own `bottle_neck_module.0` layers. Do not fall back on `strict=False`:
against an unwrapped network it loads nothing and leaves the model at its random
initialisation without raising. See `checkpoints/README.md`.

Each file is just under GitHub's 100 MB limit, so publish them as **release assets** rather
than committing them (`*.pkl` is gitignored).
