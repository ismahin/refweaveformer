from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, fields
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
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        valid = {field.name for field in fields(core.CFG)}
        for key, value in raw.items():
            if key in valid:
                if key == "CLASS_NAMES":
                    value = tuple(value)
                setattr(cfg, key, value)
    return cfg


def make_loader(pairs, cfg: core.CFG, augment: bool) -> DataLoader:
    ds = core.FabricYOLODataset(
        pairs,
        cfg.IMG_SIZE,
        cfg.NUM_CLASSES,
        augment=augment,
    )
    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=augment,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def sanity_check(model, loader, cfg: core.CFG) -> None:
    images, targets = next(iter(loader))
    images = images.to(core.device, non_blocking=True)
    targets = [t.to(core.device, non_blocking=True) for t in targets]
    model.train()
    with torch.no_grad():
        outputs = model(images)
        loss, logs = core.detection_loss(outputs, targets, cfg)
    print("Sanity image batch:", tuple(images.shape), "finite:", torch.isfinite(images).all().item())
    print("Sanity loss finite:", torch.isfinite(loss).item(), logs)
    if not torch.isfinite(loss):
        raise RuntimeError("Initial custom-model loss is not finite.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the notebook RefWeaveFormer-YOLO model locally.")
    parser.add_argument("--config", default="configs/custom_refweaveformer_4gb.yaml", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--model", choices=["v1", "v2"], help="Override model architecture.")
    parser.add_argument("--amp", choices=["true", "false"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-samples", type=int, help="Use only this many training images per epoch.")
    parser.add_argument("--max-val-samples", type=int, help="Use only this many validation images per epoch.")
    parser.add_argument("--val-every", default=1, type=int, help="Run validation every N epochs.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.img_size is not None:
        cfg.IMG_SIZE = args.img_size
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.model is not None:
        cfg.MODEL = args.model
    if args.amp is not None:
        cfg.AMP = args.amp == "true"

    out_dir = Path(cfg.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    core.seed_everything(cfg.SEED)
    core.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"GPU memory free/total GiB: {free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f}")
    print("Device:", core.device)
    print("Config:", json.dumps(asdict(cfg), indent=2))

    split_data = core.build_split_index(Path(cfg.DATA_ROOT), cfg)
    rng = random.Random(cfg.SEED)
    if args.max_train_samples is not None and args.max_train_samples > 0:
        train_pairs = list(split_data["train"])
        rng.shuffle(train_pairs)
        split_data["train"] = train_pairs[: min(args.max_train_samples, len(train_pairs))]
    if args.max_val_samples is not None and args.max_val_samples > 0:
        val_pairs = list(split_data["val"])
        rng.shuffle(val_pairs)
        split_data["val"] = val_pairs[: min(args.max_val_samples, len(val_pairs))]

    for split, pairs in split_data.items():
        print(f"{split}: {len(pairs)} images")

    train_loader = make_loader(split_data["train"], cfg, augment=True)
    val_loader = make_loader(split_data["val"], cfg, augment=False)

    model = core.build_model(cfg).to(core.device)
    core.initialize_detection_biases(model, cfg.NUM_CLASSES)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.EPOCHS),
        eta_min=cfg.MIN_LR,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.AMP and torch.cuda.is_available()))

    start_epoch = 1
    history = []
    best_val = float("inf")
    best_path = out_dir / "best_refweaveformer_yolo.pt"
    last_path = out_dir / "last_refweaveformer_yolo.pt"
    if args.resume and args.resume.exists():
        ckpt = core.load_checkpoint(model, args.resume, map_location=core.device)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print("Resumed from:", args.resume)

    sanity_check(model, train_loader, cfg)
    no_improve = 0

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        start = time.time()
        print(f"Starting epoch {epoch}/{cfg.EPOCHS}...", flush=True)
        train_logs = core.train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        should_validate = len(split_data["val"]) and (args.val_every > 0) and (
            epoch == cfg.EPOCHS or epoch % args.val_every == 0
        )
        if should_validate:
            print(f"Validating epoch {epoch}/{cfg.EPOCHS}...", flush=True)
            val_logs = core.validate_one_epoch(model, val_loader, cfg)
        else:
            val_logs = {}
        scheduler.step()

        row = {"epoch": epoch, **train_logs, **val_logs, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        core.save_history(history, out_dir)

        val_loss = val_logs.get("val_loss", train_logs.get("loss", np.inf))
        print(
            f"Epoch {epoch:03d}/{cfg.EPOCHS} | "
            f"loss={train_logs.get('loss', 0):.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={time.time() - start:.1f}s"
        )

        core.save_checkpoint(model, optimizer, epoch, last_path, cfg)
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            core.save_checkpoint(model, optimizer, epoch, best_path, cfg)
            print("Saved best:", best_path)
        else:
            no_improve += 1

        if epoch % cfg.SAVE_EVERY == 0:
            core.save_checkpoint(model, optimizer, epoch, out_dir / f"epoch_{epoch:03d}.pt", cfg)

        if no_improve >= cfg.PATIENCE:
            print(f"Early stopping after {cfg.PATIENCE} non-improving epochs.")
            break

    print("Training finished.")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    main()
