from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description="Sweep confidence/NMS thresholds for custom RefWeaveFormer models.")
    parser.add_argument("--config", default="configs/custom_refweaveformer_max.yaml", type=Path)
    parser.add_argument("--weights", default="artifacts/refweaveformer_yolo_max/best_refweaveformer_yolo.pt", type=Path)
    parser.add_argument("--model", choices=["v1", "v2"], help="Override model architecture.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--conf", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60])
    parser.add_argument("--nms", nargs="+", type=float, default=[0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.model is not None:
        cfg.MODEL = args.model
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size

    core.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = core.build_model(cfg).to(core.device)
    ckpt = core.load_checkpoint(model, args.weights, map_location=core.device)
    model.eval()
    print("Loaded:", args.weights, "epoch:", ckpt.get("epoch"))

    split_data = core.build_split_index(Path(cfg.DATA_ROOT), cfg)
    loader = make_loader(split_data[args.split], cfg)
    iou_thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]

    rows = []
    for conf in args.conf:
        for nms in args.nms:
            cfg.CONF_THRES = conf
            cfg.NMS_IOU = nms
            preds, gts = core.run_inference_on_dataset(model, loader.dataset, loader, cfg)
            metrics, _ = core.evaluate_predictions(preds, gts, cfg.NUM_CLASSES, iou_thresholds)
            row = {
                "split": args.split,
                "conf": conf,
                "nms_iou": nms,
                **metrics["overall"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    rows.sort(key=lambda r: (r["macro_F1"], r["mAP50"]), reverse=True)
    out_path = args.out or (Path(cfg.OUT_DIR) / f"{args.split}_threshold_sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Best:", json.dumps(rows[0], indent=2))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
