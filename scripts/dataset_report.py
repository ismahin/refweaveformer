from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    base = Path(data.get("path", path.parent)).expanduser()
    if not base.is_absolute():
        base = (path.parent / base).resolve()
    data["_base"] = base
    return data


def read_label_file(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    rows = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label row in {path}: {line}")
        rows.append([float(x) for x in parts])
    return rows


def inspect_split(base: Path, split_name: str, image_rel: str) -> dict:
    image_dir = base / image_rel
    label_dir = image_dir.parent / "labels"
    image_paths = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    label_paths = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []

    image_stems = {p.stem for p in image_paths}
    label_stems = {p.stem for p in label_paths}
    dimensions: Counter[str] = Counter()
    class_boxes: Counter[int] = Counter()
    empty_labels = 0
    total_boxes = 0
    bad_labels: list[str] = []

    for image_path in image_paths:
        with Image.open(image_path) as img:
            dimensions[f"{img.width}x{img.height}"] += 1

        label_path = label_dir / f"{image_path.stem}.txt"
        try:
            rows = read_label_file(label_path)
        except Exception:
            bad_labels.append(str(label_path))
            continue

        if not rows:
            empty_labels += 1
        for row in rows:
            class_id = int(row[0])
            class_boxes[class_id] += 1
            total_boxes += 1

    return {
        "split": split_name,
        "images": len(image_paths),
        "labels": len(label_paths),
        "boxes": total_boxes,
        "empty_label_files": empty_labels,
        "missing_labels": sorted(image_stems - label_stems),
        "orphan_labels": sorted(label_stems - image_stems),
        "dimensions": dict(dimensions),
        "class_boxes": {str(k): v for k, v in sorted(class_boxes.items())},
        "bad_labels": bad_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect YOLO dataset structure and labels.")
    parser.add_argument("--data", default="configs/fabric_fault.yaml", type=Path)
    parser.add_argument("--out", default="artifacts/dataset_report", type=Path)
    args = parser.parse_args()

    data = load_data_yaml(args.data)
    base: Path = data["_base"]
    args.out.mkdir(parents=True, exist_ok=True)

    split_keys = [("train", data["train"]), ("val", data["val"]), ("test", data.get("test"))]
    report = {
        "dataset_yaml": str(args.data.resolve()),
        "dataset_root": str(base),
        "names": data.get("names", {}),
        "splits": [],
    }

    for split_name, rel_path in split_keys:
        if not rel_path:
            continue
        report["splits"].append(inspect_split(base, split_name, rel_path))

    json_path = args.out / "dataset_report.json"
    csv_path = args.out / "split_summary.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "images", "labels", "boxes", "empty_label_files", "dimensions", "class_boxes"],
        )
        writer.writeheader()
        for row in report["splits"]:
            writer.writerow(
                {
                    "split": row["split"],
                    "images": row["images"],
                    "labels": row["labels"],
                    "boxes": row["boxes"],
                    "empty_label_files": row["empty_label_files"],
                    "dimensions": json.dumps(row["dimensions"]),
                    "class_boxes": json.dumps(row["class_boxes"]),
                }
            )

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    for split in report["splits"]:
        print(
            f"{split['split']}: {split['images']} images, {split['labels']} labels, "
            f"{split['boxes']} boxes, {split['empty_label_files']} empty labels"
        )


if __name__ == "__main__":
    main()
