# RefWeaveFormer-YOLO Fabric Fault Detection

Custom PyTorch implementation of the notebook model in `fabrci-fault.ipynb`.
The model is a RefWeaveFormer-YOLO style detector for fabric/textile defect localization with:

- local ConvNeXt-lite texture branch
- window-transformer global branch
- texture reference deviation modules
- cross-gated local/global fusion
- weighted BiFPN neck
- decoupled YOLO detection heads at P2/P3/P4/P5
- auxiliary defectness head

The dataset is expected in YOLO format:

```text
Dataset/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
  data.yaml
```

Large datasets, checkpoints, and training outputs are ignored by git.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Check the dataset before training:

```powershell
python scripts\dataset_report.py --data configs\fabric_fault.yaml --out artifacts\dataset_report
```

## Training

Maximum-accuracy training for a stronger GPU/PC:

```powershell
python scripts\train_custom_refweaveformer.py --config configs\custom_refweaveformer_max.yaml
```

Resume interrupted training:

```powershell
python scripts\train_custom_refweaveformer.py --config configs\custom_refweaveformer_max.yaml --resume artifacts\refweaveformer_yolo_max\last_refweaveformer_yolo.pt
```

The high-accuracy config uses:

```text
IMG_SIZE: 640
BATCH_SIZE: 8
EPOCHS: 200
NUM_WORKERS: 8
AMP: true
PATIENCE: 35
SAVE_EVERY: 10
```

If the stronger GPU still runs out of memory, lower only the batch size first:

```powershell
python scripts\train_custom_refweaveformer.py --config configs\custom_refweaveformer_max.yaml --batch-size 4
```

For laptop/debug runs:

```powershell
python scripts\train_custom_refweaveformer.py --config configs\custom_refweaveformer_fast.yaml --max-train-samples 800 --max-val-samples 100 --val-every 2
```

Training outputs are saved in the configured `OUT_DIR`, for example:

```text
artifacts/refweaveformer_yolo_max/
  best_refweaveformer_yolo.pt
  last_refweaveformer_yolo.pt
  epoch_010.pt
  training_history.csv
  loss_curve.png
  val_loss_curve.png
```

## Evaluation

Evaluate the best checkpoint on validation and test splits:

```powershell
python scripts\evaluate_custom_refweaveformer.py --config configs\custom_refweaveformer_max.yaml --weights artifacts\refweaveformer_yolo_max\best_refweaveformer_yolo.pt
```

Evaluation saves:

```text
artifacts/refweaveformer_yolo_max/
  evaluation_summary.json
  val_metrics.json
  val_overall_metrics.csv
  val_per_class_metrics.csv
  val_confusion_matrix.png
  val_pr_curve_class_0_Defect.png
  val_predictions.csv
  val_ground_truth.csv
  test_metrics.json
  test_overall_metrics.csv
  test_per_class_metrics.csv
  test_confusion_matrix.png
  test_pr_curve_class_0_Defect.png
  test_predictions.csv
  test_ground_truth.csv
  latency_estimate.json
  prediction_samples/
```

The main metrics include `mAP50`, `mAP50_95`, precision, recall, F1, AP50, and AP75.

## Configs

- `configs/custom_refweaveformer_max.yaml` - maximum-accuracy training for a high-spec PC.
- `configs/custom_refweaveformer_4gb.yaml` - conservative GTX 1650 4 GB training.
- `configs/custom_refweaveformer_fast.yaml` - quick debugging/subset training.
- `configs/fabric_fault.yaml` - YOLO dataset report config.

## Notes

The original notebook is kept as `fabrci-fault.ipynb`. The reusable implementation lives in:

```text
src/refweaveformer_yolo/core.py
scripts/train_custom_refweaveformer.py
scripts/evaluate_custom_refweaveformer.py
```
