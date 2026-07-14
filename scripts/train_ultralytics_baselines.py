from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def scalar(value: Any) -> float | None:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


def summarize_metrics(metrics: Any) -> dict:
    box = getattr(metrics, "box", None)
    result = {
        "precision": scalar(getattr(box, "mp", None)),
        "recall": scalar(getattr(box, "mr", None)),
        "mAP50": scalar(getattr(box, "map50", None)),
        "mAP50_95": scalar(getattr(box, "map", None)),
    }
    fitness = scalar(getattr(metrics, "fitness", None))
    if fitness is not None:
        result["fitness"] = fitness
    return result


def model_family(weights: str):
    try:
        from ultralytics import RTDETR, YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'ultralytics' package is required for baseline training. "
            "Install project requirements first: python -m pip install -r requirements.txt"
        ) from exc

    name = Path(weights).name.lower()
    if name.startswith("rtdetr"):
        return RTDETR
    return YOLO


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = [
        "model",
        "weights",
        "split",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "fitness",
        "best_weights",
        "save_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO/RT-DETR baselines on the same fabric-defect split."
    )
    parser.add_argument("--data", default="configs/fabric_fault.yaml", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="Ultralytics model weights to train, e.g. yolo11s.pt rtdetr-l.pt.",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or cpu.")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--project", default="artifacts/baselines", type=Path)
    parser.add_argument("--patience", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only evaluate existing runs. Expects --models to be best.pt paths or trained weight paths.",
    )
    args = parser.parse_args()

    args.project.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for weights in args.models:
        weights_path = Path(weights)
        run_name = weights_path.stem.replace(".", "_")
        print(f"\n=== Baseline: {weights} ===", flush=True)
        cls = model_family(weights)
        model = cls(weights)

        best_weights = weights
        save_dir = ""
        if not args.skip_train:
            train_result = model.train(
                data=str(args.data),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                project=str(args.project),
                name=run_name,
                exist_ok=True,
                pretrained=True,
                optimizer="AdamW",
                cos_lr=True,
                close_mosaic=15,
                patience=args.patience,
                seed=args.seed,
                plots=True,
                resume=args.resume,
            )
            save_dir = str(getattr(train_result, "save_dir", args.project / run_name))
            candidate = Path(save_dir) / "weights" / "best.pt"
            if candidate.exists():
                best_weights = str(candidate)
                model = cls(best_weights)

        model_summary = {"weights": weights, "best_weights": best_weights, "splits": {}}

        for split in ["val", "test"]:
            print(f"Evaluating {weights} on {split}...", flush=True)
            metrics = model.val(
                data=str(args.data),
                split=split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                project=str(args.project / "eval"),
                name=f"{run_name}_{split}",
                exist_ok=True,
                plots=True,
            )
            row = {
                "model": run_name,
                "weights": weights,
                "split": split,
                "best_weights": best_weights,
                "save_dir": save_dir,
                **summarize_metrics(metrics),
            }
            all_rows.append(row)
            model_summary["splits"][split] = summarize_metrics(metrics)

        summary[run_name] = model_summary

    summary_path = args.project / "baseline_summary.json"
    csv_path = args.project / "baseline_summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(all_rows, csv_path)
    print("\nSaved:", summary_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
