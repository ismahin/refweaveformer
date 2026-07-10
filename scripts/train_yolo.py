from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for fabric fault detection.")
    parser.add_argument("--config", default="configs/train_4gb.yaml", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--name")
    parser.add_argument("--model")
    parser.add_argument("--amp")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key in ["epochs", "imgsz", "batch", "workers", "device", "name", "model"]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.amp is not None:
        cfg["amp"] = str_to_bool(args.amp)

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"GPU memory free/total GiB: {free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f}")

    Path(cfg["project"]).mkdir(parents=True, exist_ok=True)
    resolved_cfg = Path(cfg["project"]) / f"{cfg['name']}_train_config.json"
    resolved_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        workers=cfg["workers"],
        device=cfg["device"],
        project=cfg["project"],
        name=cfg["name"],
        patience=cfg["patience"],
        save_period=cfg["save_period"],
        seed=cfg["seed"],
        amp=cfg["amp"],
        cache=cfg["cache"],
        cos_lr=cfg["cos_lr"],
        plots=cfg["plots"],
        exist_ok=True,
        close_mosaic=10,
        verbose=True,
    )


if __name__ == "__main__":
    main()
