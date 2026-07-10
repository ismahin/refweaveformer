from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def summarize_metrics(metrics: Any) -> dict:
    summary = {
        "results_dict": to_jsonable(getattr(metrics, "results_dict", {})),
    }
    box = getattr(metrics, "box", None)
    if box is not None:
        summary["box"] = {
            "map": to_jsonable(getattr(box, "map", None)),
            "map50": to_jsonable(getattr(box, "map50", None)),
            "map75": to_jsonable(getattr(box, "map75", None)),
            "maps": to_jsonable(getattr(box, "maps", None)),
            "mp": to_jsonable(getattr(box, "mp", None)),
            "mr": to_jsonable(getattr(box, "mr", None)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO weights on val/test splits.")
    parser.add_argument("--weights", default="runs/fabric_fault/yolo8n_4gb/weights/best.pt", type=Path)
    parser.add_argument("--data", default="configs/fabric_fault.yaml")
    parser.add_argument("--project", default="artifacts/evaluation")
    parser.add_argument("--name", default="yolo8n_4gb")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=0, type=int)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    out_root = Path(args.project) / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for split in ["val", "test"]:
        print(f"Evaluating {split}...")
        metrics = model.val(
            data=args.data,
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(out_root),
            name=split,
            exist_ok=True,
            plots=True,
            save_txt=True,
            save_conf=True,
            verbose=True,
        )
        all_metrics[split] = summarize_metrics(metrics)

    summary_path = out_root / "metrics_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
