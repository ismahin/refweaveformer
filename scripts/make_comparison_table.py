from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LITERATURE_ROWS = [
    {
        "group": "literature",
        "model": "YOLOv8 + ResNet-50 fabric detector",
        "dataset": "custom production-video fabric dataset",
        "split": "test",
        "mAP50": 0.675,
        "mAP50_95": 0.461,
        "precision": "",
        "recall": "",
        "F1": 0.610,
        "FPS": "",
        "source": "https://www.informatica.si/index.php/informatica/article/view/10031",
        "note": "Different dataset; useful context, not direct SOTA claim.",
    },
    {
        "group": "literature",
        "model": "DCFE-YOLO",
        "dataset": "fabric defect dataset used by paper",
        "split": "test",
        "mAP50": 0.894,
        "mAP50_95": "",
        "precision": "",
        "recall": "",
        "F1": "",
        "FPS": "",
        "source": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0314525",
        "note": "Different dataset; reports YOLOv8n-based improvement.",
    },
    {
        "group": "literature",
        "model": "Fab-ASLKS",
        "dataset": "Tianchi fabric defect dataset",
        "split": "test",
        "mAP50": 0.603,
        "mAP50_95": "",
        "precision": "",
        "recall": "",
        "F1": "",
        "FPS": "",
        "source": "https://arxiv.org/abs/2501.14190",
        "note": "Different dataset; reports +5 mAP50 over YOLOv8s baseline.",
    },
]


def maybe_float(value):
    if value in ("", None):
        return ""
    try:
        return float(value)
    except Exception:
        return value


def add_custom_summary(rows: list[dict], path: Path, model_name: str) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    latency_path = path.parent / "latency_estimate.json"
    fps = ""
    if latency_path.exists():
        fps = json.loads(latency_path.read_text(encoding="utf-8")).get("fps_estimate", "")
    for split, metrics in data.get("splits", {}).items():
        rows.append({
            "group": "same_dataset",
            "model": model_name,
            "dataset": "Mixed Textile Defects / local Dataset",
            "split": split,
            "mAP50": metrics.get("mAP50", ""),
            "mAP50_95": metrics.get("mAP50_95", ""),
            "precision": metrics.get("macro_precision", ""),
            "recall": metrics.get("macro_recall", ""),
            "F1": metrics.get("macro_F1", ""),
            "FPS": fps,
            "source": str(path),
            "note": "Directly comparable only with rows trained/evaluated on this same local split.",
        })


def add_baseline_csv(rows: list[dict], path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append({
                "group": "same_dataset",
                "model": raw.get("model", ""),
                "dataset": "Mixed Textile Defects / local Dataset",
                "split": raw.get("split", ""),
                "mAP50": maybe_float(raw.get("mAP50", "")),
                "mAP50_95": maybe_float(raw.get("mAP50_95", "")),
                "precision": maybe_float(raw.get("precision", "")),
                "recall": maybe_float(raw.get("recall", "")),
                "F1": "",
                "FPS": "",
                "source": raw.get("best_weights", ""),
                "note": "Ultralytics baseline trained/evaluated on the same local split.",
            })


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["group", "model", "dataset", "split", "mAP50", "mAP50_95", "precision", "recall", "F1", "FPS", "source", "note"]
    csv_path = out_dir / "paper_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "paper_comparison.md"
    lines = [
        "| Group | Model | Dataset | Split | mAP50 | mAP50-95 | Precision | Recall | F1 | FPS | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {group} | {model} | {dataset} | {split} | {mAP50} | {mAP50_95} | {precision} | {recall} | {F1} | {FPS} | {note} |".format(**r)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", csv_path)
    print("Saved:", md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready comparison tables.")
    parser.add_argument("--v1-summary", default="artifacts/refweaveformer_yolo_max/evaluation_summary.json", type=Path)
    parser.add_argument("--v2-summary", default="artifacts/refweaveformer_yolo_v2_strong/evaluation_summary.json", type=Path)
    parser.add_argument("--baseline-csv", default="artifacts/baselines/baseline_summary.csv", type=Path)
    parser.add_argument("--out-dir", default="artifacts/comparison", type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    add_custom_summary(rows, args.v1_summary, "RefWeaveFormer-YOLO v1")
    add_custom_summary(rows, args.v2_summary, "RefWeaveFormer-YOLO v2")
    add_baseline_csv(rows, args.baseline_csv)
    rows.extend(LITERATURE_ROWS)
    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()
