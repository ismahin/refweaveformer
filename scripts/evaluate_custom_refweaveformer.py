from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, fields
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.refweaveformer_yolo import core


def load_cfg(path: Path) -> core.CFG:
    cfg = core.CFG()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    valid = {field.name for field in fields(core.CFG)}
    for key, value in raw.items():
        if key in valid:
            if key == "CLASS_NAMES":
                value = tuple(value)
            setattr(cfg, key, value)
    return cfg


def make_loader(pairs, cfg: core.CFG) -> DataLoader:
    ds = core.FabricYOLODataset(pairs, cfg.IMG_SIZE, cfg.NUM_CLASSES, augment=False)
    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the custom notebook RefWeaveFormer-YOLO model.")
    parser.add_argument("--config", default="configs/custom_refweaveformer_4gb.yaml", type=Path)
    parser.add_argument("--weights", default="artifacts/refweaveformer_yolo/best_refweaveformer_yolo.pt", type=Path)
    parser.add_argument("--batch-size", type=int, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader worker count.")
    parser.add_argument("--model", choices=["v1", "v2"], help="Override model architecture.")
    parser.add_argument("--amp", choices=["true", "false"], help="Override mixed precision inference.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers
    if args.model is not None:
        cfg.MODEL = args.model
    if args.amp is not None:
        cfg.AMP = args.amp == "true"

    out_dir = Path(cfg.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    core.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = core.build_model(cfg).to(core.device)
    ckpt = core.load_checkpoint(model, args.weights, map_location=core.device)
    model.eval()
    print("Loaded:", args.weights, "epoch:", ckpt.get("epoch"))
    print("Device:", core.device, "| eval batch size:", cfg.BATCH_SIZE, "| AMP:", cfg.AMP)

    split_data = core.build_split_index(Path(cfg.DATA_ROOT), cfg)
    iou_thresholds = [round(x, 2) for x in list(__import__("numpy").arange(0.50, 0.96, 0.05))]
    summary = {"config": asdict(cfg), "weights": str(args.weights), "splits": {}}

    for split_name in ["val", "test"]:
        pairs = split_data.get(split_name, [])
        if not pairs:
            print("No split data, skipping:", split_name)
            continue
        loader = make_loader(pairs, cfg)
        dataset = loader.dataset
        print(f"Evaluating {split_name}: {len(dataset)} images")
        preds, gts = core.run_inference_on_dataset(model, dataset, loader, cfg)
        metrics, pr_data = core.evaluate_predictions(preds, gts, cfg.NUM_CLASSES, iou_thresholds)
        core.save_metrics(metrics, cfg.CLASS_NAMES, out_dir, prefix=split_name)
        core.save_pr_curves(pr_data, cfg.CLASS_NAMES, out_dir, prefix=split_name)
        mat = core.compute_confusion_matrix(preds, gts, cfg.NUM_CLASSES, iou_thr=0.5)
        core.save_confusion_matrix(mat, cfg.CLASS_NAMES, out_dir, prefix=split_name)
        core.save_predictions_csv(preds, gts, out_dir, prefix=split_name)
        summary["splits"][split_name] = metrics["overall"]
        print(split_name, json.dumps(metrics["overall"], indent=2))

    core.measure_latency(model, cfg.IMG_SIZE, out_dir, amp=cfg.AMP)

    sample_dir = out_dir / "prediction_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    test_pairs = split_data.get("test", [])
    for i, (img_path, _) in enumerate(random.sample(test_pairs, min(8, len(test_pairs)))):
        image_np, pred = core.predict_single_image(model, img_path, cfg)
        core.plot_prediction(
            image_np,
            pred,
            cfg.CLASS_NAMES,
            save_path=sample_dir / f"sample_prediction_{i + 1}.png",
        )

    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved evaluation summary:", summary_path)


if __name__ == "__main__":
    main()
