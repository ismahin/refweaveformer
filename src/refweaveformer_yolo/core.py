from __future__ import annotations
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import yaml
except Exception:
    yaml = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

@dataclass
class CFG:
    DATA_ROOT: str = "/kaggle/input/datasets/mahinshikder/fabric-fault"
    OUT_DIR: str = "/kaggle/working/refweaveformer_yolo_results"

    IMG_SIZE: int = 640
    NUM_CLASSES: int = 1
    CLASS_NAMES: Tuple[str, ...] = ("scratch",)

    EPOCHS: int = 100
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2

    LR: float = 5e-5
    MIN_LR: float = 1e-6
    WEIGHT_DECAY: float = 1e-4

    CONF_THRES: float = 0.25
    NMS_IOU: float = 0.50
    MAX_DET: int = 300

    VAL_FRAC: float = 0.15
    TEST_FRAC: float = 0.15
    SEED: int = 42
    AMP: bool = False
    PATIENCE: int = 25
    SAVE_EVERY: int = 10

    BOX_WEIGHT: float = 7.5
    OBJ_WEIGHT: float = 1.0
    CLS_WEIGHT: float = 1.5
    AUX_WEIGHT: float = 0.35


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def read_data_yaml(data_root: Path):
    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists() or yaml is None:
        return None
    with open(data_yaml, "r") as f:
        return yaml.safe_load(f)


def list_images(img_dir: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(img_dir.rglob(f"*{ext}"))
        files.extend(img_dir.rglob(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def find_split_dirs(data_root: Path, split: str):
    aliases = [split]
    if split == "val":
        aliases.append("valid")

    candidates = []
    for s in aliases:
        candidates.extend([
            (data_root / "images" / s, data_root / "labels" / s),
            (data_root / s / "images", data_root / s / "labels"),
        ])

    for img_dir, lbl_dir in candidates:
        if img_dir.exists() and lbl_dir.exists():
            return img_dir, lbl_dir
    return None, None


def label_path_from_image(img_path: Path, img_dir: Path, lbl_dir: Path):
    rel = img_path.relative_to(img_dir)
    return (lbl_dir / rel).with_suffix(".txt")


def load_yolo_labels(label_path: Path, num_classes: int):
    """
    Robust YOLO label reader.
    Removes invalid, NaN, infinite, zero-area, and out-of-range labels.
    """
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            vals = np.array([float(x) for x in parts[:5]], dtype=np.float32)
        except Exception:
            continue

        if not np.all(np.isfinite(vals)):
            continue

        cls = int(vals[0])
        xc, yc, bw, bh = vals[1:5]

        if cls < 0 or cls >= num_classes:
            continue

        if bw <= 1e-6 or bh <= 1e-6:
            continue

        xc = float(np.clip(xc, 0.0, 1.0))
        yc = float(np.clip(yc, 0.0, 1.0))
        bw = float(np.clip(bw, 1e-5, 1.0))
        bh = float(np.clip(bh, 1e-5, 1.0))

        rows.append([cls, xc, yc, bw, bh])

    if len(rows) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)


def build_split_index(data_root: Path, cfg: CFG):
    split_data = {}
    found_any = False

    for split in ["train", "val", "test"]:
        img_dir, lbl_dir = find_split_dirs(data_root, split)
        if img_dir is not None:
            found_any = True
            images = list_images(img_dir)
            split_data[split] = [(p, label_path_from_image(p, img_dir, lbl_dir)) for p in images]
        else:
            split_data[split] = []

    if found_any and len(split_data["train"]) > 0:
        rng = np.random.default_rng(cfg.SEED)
        rng.shuffle(split_data["train"])

        if len(split_data["val"]) == 0 or len(split_data["test"]) == 0:
            train_all = split_data["train"]
            n = len(train_all)
            n_test = int(n * cfg.TEST_FRAC) if len(split_data["test"]) == 0 else 0
            n_val = int(n * cfg.VAL_FRAC) if len(split_data["val"]) == 0 else 0

            if len(split_data["test"]) == 0:
                split_data["test"] = train_all[:n_test]
            if len(split_data["val"]) == 0:
                split_data["val"] = train_all[n_test:n_test+n_val]
            split_data["train"] = train_all[n_test+n_val:]

        return split_data

    img_dir = data_root / "images"
    lbl_dir = data_root / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(
            "Could not find YOLO dataset. Expected images/train + labels/train, "
            "train/images + train/labels, or images + labels."
        )

    pairs = [(p, label_path_from_image(p, img_dir, lbl_dir)) for p in list_images(img_dir)]
    rng = np.random.default_rng(cfg.SEED)
    rng.shuffle(pairs)

    n = len(pairs)
    n_test = int(n * cfg.TEST_FRAC)
    n_val = int(n * cfg.VAL_FRAC)

    return {
        "test": pairs[:n_test],
        "val": pairs[n_test:n_test+n_val],
        "train": pairs[n_test+n_val:],
    }


def resize_and_load_image(path: Path, img_size: int):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def apply_train_augment(image: np.ndarray, boxes: np.ndarray):
    img = image
    b = boxes.copy()

    if random.random() < 0.5:
        img = img[:, ::-1, :]
        if len(b):
            b[:, 1] = 1.0 - b[:, 1]

    if random.random() < 0.15:
        img = img[::-1, :, :]
        if len(b):
            b[:, 2] = 1.0 - b[:, 2]

    if random.random() < 0.70:
        img = np.clip(img * random.uniform(0.65, 1.35), 0.0, 1.0)

    if random.random() < 0.70:
        mean = img.mean(axis=(0, 1), keepdims=True)
        factor = random.uniform(0.70, 1.40)
        img = np.clip((img - mean) * factor + mean, 0.0, 1.0)

    if random.random() < 0.35:
        sigma = random.uniform(0.005, 0.035)
        img = np.clip(img + np.random.normal(0, sigma, img.shape), 0.0, 1.0)

    if random.random() < 0.15:
        pil = Image.fromarray(np.uint8(img * 255.0))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        img = np.asarray(pil).astype(np.float32) / 255.0

    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    b = np.nan_to_num(b.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return img, b


def apply_corruption(image: np.ndarray, corruption: str):
    img = image.copy()

    if corruption == "clean":
        return img
    if corruption == "low_light":
        return np.clip(img * 0.55, 0.0, 1.0)
    if corruption == "high_light":
        return np.clip(img * 1.45, 0.0, 1.0)
    if corruption == "contrast_shift":
        mean = img.mean(axis=(0, 1), keepdims=True)
        return np.clip((img - mean) * 1.65 + mean, 0.0, 1.0)
    if corruption == "gaussian_noise":
        return np.clip(img + np.random.normal(0, 0.045, img.shape), 0.0, 1.0).astype(np.float32)
    if corruption == "blur":
        pil = Image.fromarray(np.uint8(img * 255.0))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=1.4))
        return np.asarray(pil).astype(np.float32) / 255.0

    return np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)


def choose_scale_for_box(w, h):
    max_side = max(w, h)
    area = w * h

    if max_side < 0.070 or area < 0.0020:
        return 0
    if max_side < 0.160 or area < 0.0120:
        return 1
    if max_side < 0.340 or area < 0.0600:
        return 2
    return 3


def assign_positive(target, cls, xc, yc, bw, bh, num_classes, radius=1):
    gh, gw = target.shape[:2]
    gx0 = int(np.floor(xc * gw))
    gy0 = int(np.floor(yc * gh))

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            gx = gx0 + dx
            gy = gy0 + dy
            if gx < 0 or gx >= gw or gy < 0 or gy >= gh:
                continue

            new_area = bw * bh
            old_obj = target[gy, gx, 4]
            old_area = target[gy, gx, 2] * target[gy, gx, 3] if old_obj > 0 else 1e9

            if old_obj == 0 or new_area <= old_area:
                target[gy, gx, 0:4] = [xc, yc, bw, bh]
                target[gy, gx, 4] = 1.0
                target[gy, gx, 5:] = 0.0
                target[gy, gx, 5 + cls] = 1.0


def build_targets(boxes, img_size, num_classes):
    grids = [img_size // 4, img_size // 8, img_size // 16, img_size // 32]
    channels = 5 + num_classes

    targets = [np.zeros((g, g, channels), dtype=np.float32) for g in grids]

    aux_g = img_size // 4
    aux_mask = np.zeros((aux_g, aux_g, 1), dtype=np.float32)

    for row in boxes:
        if not np.all(np.isfinite(row)):
            continue

        cls = int(row[0])
        xc, yc, bw, bh = row[1:5]
        if bw <= 0 or bh <= 0:
            continue

        scale_idx = choose_scale_for_box(bw, bh)
        aspect = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))

        scale_indices = [scale_idx]
        if aspect > 5.0 and scale_idx != 0:
            scale_indices.append(0)

        for sidx in sorted(set(scale_indices)):
            radius = 1 if sidx <= 1 else 0
            assign_positive(targets[sidx], cls, xc, yc, bw, bh, num_classes, radius)

        x1 = int(np.floor((xc - bw / 2) * aux_g))
        y1 = int(np.floor((yc - bh / 2) * aux_g))
        x2 = int(np.ceil((xc + bw / 2) * aux_g))
        y2 = int(np.ceil((yc + bh / 2) * aux_g))

        x1 = np.clip(x1, 0, aux_g - 1)
        y1 = np.clip(y1, 0, aux_g - 1)
        x2 = np.clip(x2, x1 + 1, aux_g)
        y2 = np.clip(y2, y1 + 1, aux_g)

        aux_mask[y1:y2, x1:x2, 0] = 1.0

    return targets, aux_mask.astype(np.float32)


class FabricYOLODataset(Dataset):
    def __init__(self, pairs, img_size, num_classes, augment=False, corruption="clean"):
        self.pairs = list(pairs)
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
        self.corruption = corruption

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        image = resize_and_load_image(img_path, self.img_size)
        boxes = load_yolo_labels(lbl_path, self.num_classes)

        if self.augment:
            image, boxes = apply_train_augment(image, boxes)
        else:
            image = apply_corruption(image, self.corruption)

        targets, aux_mask = build_targets(boxes, self.img_size, self.num_classes)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        target_tensors = [torch.from_numpy(t).float() for t in targets]
        aux_mask = torch.from_numpy(aux_mask).float()

        return image, (*target_tensors, aux_mask)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, groups=1, act=True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, ch, mlp_ratio=4, drop=0.0):
        super().__init__()
        hidden = ch * mlp_ratio
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.norm = nn.GroupNorm(1, ch)
        self.pw1 = nn.Conv2d(ch, hidden, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.pw2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw2(x)
        x = self.drop(x)
        return identity + x


class CSPConvNeXtStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2):
        super().__init__()
        self.in_conv = ConvBNAct(in_ch, out_ch, k=1, s=1)
        hidden = out_ch // 2
        self.blocks = nn.Sequential(*[ConvNeXtLiteBlock(hidden) for _ in range(depth)])
        self.out_conv = ConvBNAct(out_ch, out_ch, k=1, s=1)

    def forward(self, x):
        x = self.in_conv(x)
        a, b = torch.chunk(x, 2, dim=1)
        b = self.blocks(b)
        return self.out_conv(torch.cat([a, b], dim=1))


class WindowTransformerBlock(nn.Module):
    """
    Safe dynamic-padded window Transformer block.

    This version avoids nn.MultiheadAttention and PyTorch fused SDPA kernels.
    It uses explicit QKV projection + matmul attention, which is more stable
    on Kaggle dual-GPU DataParallel with T4 GPUs.
    """
    def __init__(self, dim, heads=4, window_size=8, mlp_ratio=2.0, drop=0.05):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} must be divisible by heads={heads}"

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: B,C,H,W
        x = x.contiguous()
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h)).contiguous()

        Hp, Wp = x.shape[-2], x.shape[-1]

        # B,C,H,W -> B,H,W,C
        x = x.permute(0, 2, 3, 1).contiguous()

        # Window partition: [num_windows*B, ws*ws, C]
        x_windows = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_windows = x_windows.view(-1, ws * ws, C).contiguous()

        shortcut = x_windows

        y = self.norm1(x_windows)

        # Manual QKV attention. Avoids torch SDPA fused kernels.
        qkv = self.qkv(y)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn.float(), dim=-1).type_as(q)
        attn = self.attn_drop(attn)

        y = (attn @ v).transpose(1, 2).reshape(x_windows.shape[0], x_windows.shape[1], C)
        y = self.proj(y)
        y = self.proj_drop(y)

        x_windows = shortcut + y
        x_windows = x_windows + self.mlp(self.norm2(x_windows))

        # Reverse windows
        x = x_windows.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)
        x = x[:, :H, :W, :].contiguous()

        return x.permute(0, 3, 1, 2).contiguous()


class TransformerStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth=1, heads=4, window_size=8):
        super().__init__()
        self.proj = ConvBNAct(in_ch, out_ch, k=1, s=1)
        self.blocks = nn.Sequential(*[
            WindowTransformerBlock(out_ch, heads=heads, window_size=window_size)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(self.proj(x))


class TextureReferenceDeviationModule(nn.Module):
    def __init__(self, ch, pool_size=9):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.diff_conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        ref = self.pool(x)
        diff = torch.abs(x - ref)
        diff = self.diff_conv(diff)
        gate = self.gate(torch.cat([x, diff], dim=1))
        return self.out(x * gate + diff)


class CrossGatedFusion(nn.Module):
    def __init__(self, local_ch, global_ch, out_ch):
        super().__init__()
        self.local_proj = ConvBNAct(local_ch, out_ch, k=1)
        self.global_proj = ConvBNAct(global_ch, out_ch, k=1)
        self.dev = TextureReferenceDeviationModule(out_ch)

        self.local_gate = nn.Sequential(nn.Conv2d(out_ch * 3, out_ch, 1), nn.Sigmoid())
        self.global_gate = nn.Sequential(nn.Conv2d(out_ch * 3, out_ch, 1), nn.Sigmoid())
        self.dev_gate = nn.Sequential(nn.Conv2d(out_ch * 3, out_ch, 1), nn.Sigmoid())

        self.out = ConvBNAct(out_ch, out_ch, k=3)

    def forward(self, local_feat, global_feat):
        l = self.local_proj(local_feat)
        g = self.global_proj(global_feat)
        d = self.dev(l)

        cat = torch.cat([l, g, d], dim=1)

        fused = (
            l * self.local_gate(cat) +
            g * self.global_gate(cat) +
            d * self.dev_gate(cat)
        )

        return self.out(fused)


class WeightedAdd(nn.Module):
    def __init__(self, n_inputs, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = 0.0
        for i, x in enumerate(inputs):
            out = out + w[i] * x
        return out


class WeightedBiFPN(nn.Module):
    def __init__(self, in_channels, out_ch=160):
        super().__init__()
        c2, c3, c4, c5 = in_channels

        self.p2_in = ConvBNAct(c2, out_ch, k=1)
        self.p3_in = ConvBNAct(c3, out_ch, k=1)
        self.p4_in = ConvBNAct(c4, out_ch, k=1)
        self.p5_in = ConvBNAct(c5, out_ch, k=1)

        self.w_p4_td = WeightedAdd(2)
        self.w_p3_td = WeightedAdd(2)
        self.w_p2_td = WeightedAdd(2)

        self.w_p3_out = WeightedAdd(3)
        self.w_p4_out = WeightedAdd(3)
        self.w_p5_out = WeightedAdd(2)

        self.p4_td_conv = ConvBNAct(out_ch, out_ch, k=3)
        self.p3_td_conv = ConvBNAct(out_ch, out_ch, k=3)
        self.p2_td_conv = ConvBNAct(out_ch, out_ch, k=3)

        self.p3_out_conv = ConvBNAct(out_ch, out_ch, k=3)
        self.p4_out_conv = ConvBNAct(out_ch, out_ch, k=3)
        self.p5_out_conv = ConvBNAct(out_ch, out_ch, k=3)

    def forward(self, feats):
        p2, p3, p4, p5 = feats

        p2 = self.p2_in(p2)
        p3 = self.p3_in(p3)
        p4 = self.p4_in(p4)
        p5 = self.p5_in(p5)

        p5_td = p5

        p4_td = self.w_p4_td([p4, F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest")])
        p4_td = self.p4_td_conv(p4_td)

        p3_td = self.w_p3_td([p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")])
        p3_td = self.p3_td_conv(p3_td)

        p2_td = self.w_p2_td([p2, F.interpolate(p3_td, size=p2.shape[-2:], mode="nearest")])
        p2_td = self.p2_td_conv(p2_td)

        p2_out = p2_td

        p3_out = self.w_p3_out([p3, p3_td, F.max_pool2d(p2_out, 2, 2)])
        p3_out = self.p3_out_conv(p3_out)

        p4_out = self.w_p4_out([p4, p4_td, F.max_pool2d(p3_out, 2, 2)])
        p4_out = self.p4_out_conv(p4_out)

        p5_out = self.w_p5_out([p5, F.max_pool2d(p4_out, 2, 2)])
        p5_out = self.p5_out_conv(p5_out)

        return [p2_out, p3_out, p4_out, p5_out]


class DecoupledYOLOHead(nn.Module):
    def __init__(self, in_ch, num_classes, hidden=160):
        super().__init__()
        self.box = nn.Sequential(
            ConvBNAct(in_ch, hidden, k=3),
            ConvBNAct(hidden, hidden, k=3),
            nn.Conv2d(hidden, 4, 1),
        )
        self.cls = nn.Sequential(
            ConvBNAct(in_ch, hidden, k=3),
            ConvBNAct(hidden, hidden, k=3),
        )
        self.obj = nn.Conv2d(hidden, 1, 1)
        self.logits = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, x):
        x = x.contiguous()
        box = self.box(x)
        c = self.cls(x)
        obj = self.obj(c)
        logits = self.logits(c)
        return torch.cat([box, obj, logits], dim=1)


class AuxiliaryDefectnessHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_ch, 96, k=3),
            ConvBNAct(96, 64, k=3),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return self.head(x)


class RefWeaveFormerYOLO(nn.Module):
    def __init__(self, num_classes=1, fpn_ch=160):
        super().__init__()

        self.stem1 = ConvBNAct(3, 32, k=3, s=2)
        self.stem2 = ConvBNAct(32, 64, k=3, s=2)

        self.local_p2 = CSPConvNeXtStage(64, 128, depth=2)
        self.local_p3_down = ConvBNAct(128, 256, k=3, s=2)
        self.local_p3 = CSPConvNeXtStage(256, 256, depth=3)

        self.local_p4_down = ConvBNAct(256, 384, k=3, s=2)
        self.local_p4 = CSPConvNeXtStage(384, 384, depth=4)

        self.local_p5_down = ConvBNAct(384, 512, k=3, s=2)
        self.local_p5 = CSPConvNeXtStage(512, 512, depth=3)

        self.global_p2 = TransformerStage(64, 128, depth=1, heads=4, window_size=8)

        self.global_p3_down = ConvBNAct(128, 256, k=3, s=2)
        self.global_p3 = TransformerStage(256, 256, depth=1, heads=4, window_size=8)

        self.global_p4_down = ConvBNAct(256, 384, k=3, s=2)
        self.global_p4 = TransformerStage(384, 384, depth=2, heads=6, window_size=8)

        self.global_p5_down = ConvBNAct(384, 512, k=3, s=2)
        self.global_p5 = TransformerStage(512, 512, depth=2, heads=8, window_size=5)

        self.fuse_p2 = CrossGatedFusion(128, 128, 128)
        self.fuse_p3 = CrossGatedFusion(256, 256, 256)
        self.fuse_p4 = CrossGatedFusion(384, 384, 384)
        self.fuse_p5 = CrossGatedFusion(512, 512, 512)

        self.neck = WeightedBiFPN([128, 256, 384, 512], out_ch=fpn_ch)

        self.head_p2 = DecoupledYOLOHead(fpn_ch, num_classes, hidden=fpn_ch)
        self.head_p3 = DecoupledYOLOHead(fpn_ch, num_classes, hidden=fpn_ch)
        self.head_p4 = DecoupledYOLOHead(fpn_ch, num_classes, hidden=fpn_ch)
        self.head_p5 = DecoupledYOLOHead(fpn_ch, num_classes, hidden=fpn_ch)

        self.aux_head = AuxiliaryDefectnessHead(fpn_ch)

    def forward(self, x):
        x = self.stem1(x)
        x = self.stem2(x)

        l2 = self.local_p2(x)
        l3 = self.local_p3(self.local_p3_down(l2))
        l4 = self.local_p4(self.local_p4_down(l3))
        l5 = self.local_p5(self.local_p5_down(l4))

        g2 = self.global_p2(x)
        g3 = self.global_p3(self.global_p3_down(g2))
        g4 = self.global_p4(self.global_p4_down(g3))
        g5 = self.global_p5(self.global_p5_down(g4))

        f2 = self.fuse_p2(l2, g2)
        f3 = self.fuse_p3(l3, g3)
        f4 = self.fuse_p4(l4, g4)
        f5 = self.fuse_p5(l5, g5)

        p2, p3, p4, p5 = self.neck([f2, f3, f4, f5])
        p2, p3, p4, p5 = p2.contiguous(), p3.contiguous(), p4.contiguous(), p5.contiguous()

        out_p2 = self.head_p2(p2)
        out_p3 = self.head_p3(p3)
        out_p4 = self.head_p4(p4)
        out_p5 = self.head_p5(p5)
        aux = self.aux_head(p2)

        return [out_p2, out_p3, out_p4, out_p5, aux]


def initialize_detection_biases(model, num_classes):
    """
    Stabilizes early YOLO training by preventing dense object predictions
    at the beginning of training.
    """
    m = model.module if isinstance(model, nn.DataParallel) else model

    for name, module in m.named_modules():
        if isinstance(module, DecoupledYOLOHead):
            nn.init.normal_(module.box[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(module.box[-1].bias, 0.0)
            nn.init.constant_(module.obj.bias, -4.5)
            nn.init.constant_(module.logits.bias, -2.0)

    if hasattr(m, "aux_head"):
        last = m.aux_head.head[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.constant_(last.bias, -2.0)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def decode_raw_yolo(pred):
    """
    Numerically stable YOLO decoding.
    Loss is computed in float32 even when AMP is enabled.
    """
    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=10.0, neginf=-10.0)

    B, Ch, H, W = pred.shape
    pred = pred.permute(0, 2, 3, 1).contiguous()

    raw_xy = torch.clamp(pred[..., 0:2], -10.0, 10.0)
    raw_wh = torch.clamp(pred[..., 2:4], -10.0, 10.0)
    obj_logit = torch.clamp(pred[..., 4:5], -30.0, 30.0)
    cls_logit = torch.clamp(pred[..., 5:], -30.0, 30.0)

    yv, xv = torch.meshgrid(
        torch.arange(H, device=pred.device, dtype=pred.dtype),
        torch.arange(W, device=pred.device, dtype=pred.dtype),
        indexing="ij",
    )

    grid = torch.stack([xv, yv], dim=-1).view(1, H, W, 2)
    scale = torch.tensor([W, H], device=pred.device, dtype=pred.dtype).view(1, 1, 1, 2)

    xy = (torch.sigmoid(raw_xy) + grid) / scale
    wh = torch.sigmoid(raw_wh).clamp(1e-4, 0.98)

    boxes_xywh = torch.cat([xy, wh], dim=-1)
    boxes_xywh = torch.nan_to_num(boxes_xywh, nan=0.5, posinf=0.98, neginf=1e-4)

    return boxes_xywh, obj_logit, cls_logit


def xywh_to_xyxy(boxes):
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)


def bbox_iou_xyxy(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)

    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = torch.clamp(b1_x2 - b1_x1, min=0) * torch.clamp(b1_y2 - b1_y1, min=0)
    area2 = torch.clamp(b2_x2 - b2_x1, min=0) * torch.clamp(b2_y2 - b2_y1, min=0)

    return inter / (area1 + area2 - inter + eps)


def ciou_loss_xywh(pred_xywh, true_xywh, eps=1e-6):
    pred_xywh = torch.nan_to_num(pred_xywh.float(), nan=0.5, posinf=0.98, neginf=1e-4)
    true_xywh = torch.nan_to_num(true_xywh.float(), nan=0.5, posinf=1.0, neginf=1e-4)

    pred_xywh = torch.clamp(pred_xywh, 1e-4, 0.999)
    true_xywh = torch.clamp(true_xywh, 1e-4, 0.999)

    pred = xywh_to_xyxy(pred_xywh)
    true = xywh_to_xyxy(true_xywh)

    iou = bbox_iou_xyxy(pred, true, eps=eps).clamp(0.0, 1.0)

    p_x1, p_y1, p_x2, p_y2 = pred.unbind(-1)
    t_x1, t_y1, t_x2, t_y2 = true.unbind(-1)

    p_cx = (p_x1 + p_x2) / 2
    p_cy = (p_y1 + p_y2) / 2
    t_cx = (t_x1 + t_x2) / 2
    t_cy = (t_y1 + t_y2) / 2

    center_dist = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

    c_x1 = torch.minimum(p_x1, t_x1)
    c_y1 = torch.minimum(p_y1, t_y1)
    c_x2 = torch.maximum(p_x2, t_x2)
    c_y2 = torch.maximum(p_y2, t_y2)

    c_diag = ((c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2).clamp(min=eps)

    p_w = torch.clamp(p_x2 - p_x1, min=eps)
    p_h = torch.clamp(p_y2 - p_y1, min=eps)
    t_w = torch.clamp(t_x2 - t_x1, min=eps)
    t_h = torch.clamp(t_y2 - t_y1, min=eps)

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(t_w / t_h) - torch.atan(p_w / p_h), 2)
    alpha = v / torch.clamp(1 - iou + v, min=eps)

    ciou = iou - center_dist / c_diag - alpha * v
    loss = 1 - ciou

    return torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=0.0)


def binary_focal_loss_with_logits(target, logits, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = target * prob + (1 - target) * (1 - prob)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    return alpha_factor * (1 - p_t).pow(gamma) * bce


def dice_loss_with_logits(target, logits, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    union = prob.sum() + target.sum()
    return 1 - (2 * inter + eps) / (union + eps)


def detection_loss(outputs, targets, cfg: CFG):
    """
    Stable detection loss.

    Important fix:
    CIoU is computed only on positive cells. This avoids NaN propagation
    from background cells, because NaN * 0 is still NaN in PyTorch.
    """
    preds = [o.float() for o in outputs[:4]]
    aux_pred = outputs[4].float()

    det_targets = [t.float() for t in targets[:4]]
    aux_target = targets[4].float()

    total_box = aux_pred.new_tensor(0.0)
    total_obj = aux_pred.new_tensor(0.0)
    total_cls = aux_pred.new_tensor(0.0)

    scale_weights = [1.40, 1.15, 1.00, 0.85]

    for i, (pred, target) in enumerate(zip(preds, det_targets)):
        target = torch.nan_to_num(target.to(pred.device), nan=0.0, posinf=1.0, neginf=0.0)

        pred_xywh, obj_logit, cls_logit = decode_raw_yolo(pred)

        true_xywh = target[..., 0:4]
        obj_true = target[..., 4:5].clamp(0.0, 1.0)
        cls_true = target[..., 5:].clamp(0.0, 1.0)

        pos_mask = obj_true.squeeze(-1) > 0.5

        if pos_mask.any():
            box_loss = ciou_loss_xywh(pred_xywh[pos_mask], true_xywh[pos_mask]).mean()
            cls_loss_map = binary_focal_loss_with_logits(
                cls_true[pos_mask],
                cls_logit[pos_mask],
                alpha=0.25,
                gamma=2.0
            )
            cls_loss = cls_loss_map.mean()
        else:
            box_loss = pred.new_tensor(0.0)
            cls_loss = pred.new_tensor(0.0)

        obj_loss = binary_focal_loss_with_logits(
            obj_true,
            obj_logit,
            alpha=0.25,
            gamma=2.0
        ).mean()

        box_loss = torch.nan_to_num(box_loss, nan=0.0, posinf=10.0, neginf=0.0)
        obj_loss = torch.nan_to_num(obj_loss, nan=0.0, posinf=10.0, neginf=0.0)
        cls_loss = torch.nan_to_num(cls_loss, nan=0.0, posinf=10.0, neginf=0.0)

        sw = scale_weights[i]
        total_box = total_box + sw * box_loss
        total_obj = total_obj + sw * obj_loss
        total_cls = total_cls + sw * cls_loss

    aux_target = torch.nan_to_num(aux_target.to(aux_pred.device), nan=0.0, posinf=1.0, neginf=0.0)
    aux_target = aux_target.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)

    aux_logits = torch.clamp(aux_pred, -30.0, 30.0)
    aux_bce = F.binary_cross_entropy_with_logits(aux_logits, aux_target)
    aux_dice = dice_loss_with_logits(aux_target, aux_logits)
    aux_loss = torch.nan_to_num(aux_bce + aux_dice, nan=0.0, posinf=10.0, neginf=0.0)

    total = (
        cfg.BOX_WEIGHT * total_box +
        cfg.OBJ_WEIGHT * total_obj +
        cfg.CLS_WEIGHT * total_cls +
        cfg.AUX_WEIGHT * aux_loss
    )

    total = torch.nan_to_num(total, nan=0.0, posinf=100.0, neginf=0.0)

    logs = {
        "loss": float(total.detach().cpu()),
        "box_loss": float(total_box.detach().cpu()),
        "obj_loss": float(total_obj.detach().cpu()),
        "cls_loss": float(total_cls.detach().cpu()),
        "aux_loss": float(aux_loss.detach().cpu()),
    }

    return total, logs


def save_checkpoint(model, optimizer, epoch, path, cfg=None):
    payload = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if cfg is not None:
        payload["cfg"] = asdict(cfg)
    torch.save({
        **payload,
    }, path)


def load_checkpoint(model, path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
    return ckpt


def train_one_epoch(model, loader, optimizer, scaler, cfg):
    model.train()
    meters = {"loss": [], "box_loss": [], "obj_loss": [], "cls_loss": [], "aux_loss": []}
    start_time = time.time()
    total_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in targets]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.AMP and torch.cuda.is_available())):
            outputs = model(images)
            loss, logs = detection_loss(outputs, targets, cfg)

        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss detected; skipping this batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        for k in meters:
            meters[k].append(logs[k])

        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
            elapsed = time.time() - start_time
            avg_sec = elapsed / max(batch_idx, 1)
            remaining = avg_sec * max(total_batches - batch_idx, 0)
            print(
                f"  train batch {batch_idx}/{total_batches} | "
                f"loss={logs['loss']:.4f} | "
                f"avg={avg_sec:.2f}s/batch | "
                f"eta={remaining / 60:.1f}m",
                flush=True,
            )

    return {k: float(np.mean(v)) for k, v in meters.items()}


def validate_one_epoch(model, loader, cfg):
    model.eval()
    meters = {"val_loss": [], "val_box_loss": [], "val_obj_loss": [], "val_cls_loss": [], "val_aux_loss": []}
    total_batches = len(loader)

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in targets]

            with torch.cuda.amp.autocast(enabled=(cfg.AMP and torch.cuda.is_available())):
                outputs = model(images)
                loss, logs = detection_loss(outputs, targets, cfg)

            meters["val_loss"].append(logs["loss"])
            meters["val_box_loss"].append(logs["box_loss"])
            meters["val_obj_loss"].append(logs["obj_loss"])
            meters["val_cls_loss"].append(logs["cls_loss"])
            meters["val_aux_loss"].append(logs["aux_loss"])

            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
                print(
                    f"  val batch {batch_idx}/{total_batches} | "
                    f"val_loss={logs['loss']:.4f}",
                    flush=True,
                )

    return {k: float(np.mean(v)) if len(v) else 0.0 for k, v in meters.items()}


def save_history(history, out_dir):
    out_dir = Path(out_dir)
    if not history:
        return

    keys = list(history[0].keys())
    with open(out_dir / "training_history.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for row in history:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    for key in ["loss", "val_loss", "box_loss", "obj_loss", "cls_loss", "aux_loss"]:
        if key not in history[0]:
            continue
        plt.figure(figsize=(7, 5))
        plt.plot([r["epoch"] for r in history], [r.get(key, np.nan) for r in history], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(key)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}_curve.png", dpi=300)
        plt.close()


def np_xywh_to_xyxy(boxes):
    if boxes.size == 0:
        return boxes.reshape(0, 4)

    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)
    return np.clip(out, 0.0, 1.0)


def iou_numpy(box, boxes):
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter + 1e-7)


def nms_numpy(boxes, scores, iou_thres=0.5, max_det=300):
    if len(boxes) == 0:
        return []

    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0 and len(keep) < max_det:
        current = idxs[0]
        keep.append(current)

        if len(idxs) == 1:
            break

        ious = iou_numpy(boxes[current], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_thres]

    return keep


def decode_predictions(outputs, cfg):
    preds = outputs[:4]
    B = preds[0].shape[0]

    all_boxes = [[] for _ in range(B)]
    all_scores = [[] for _ in range(B)]
    all_classes = [[] for _ in range(B)]

    for pred in preds:
        boxes_xywh, obj_logit, cls_logit = decode_raw_yolo(pred)

        boxes_xywh = boxes_xywh.detach().float().cpu().numpy()
        obj = torch.sigmoid(obj_logit).detach().float().cpu().numpy()
        cls_prob = torch.sigmoid(cls_logit).detach().float().cpu().numpy()

        b, h, w, _ = boxes_xywh.shape
        boxes_flat = boxes_xywh.reshape(b, -1, 4)
        obj_flat = obj.reshape(b, -1, 1)
        cls_flat = cls_prob.reshape(b, -1, cfg.NUM_CLASSES)
        scores_all = obj_flat * cls_flat

        for bi in range(b):
            for c in range(cfg.NUM_CLASSES):
                scores = scores_all[bi, :, c]
                mask = scores >= cfg.CONF_THRES

                if not np.any(mask):
                    continue

                boxes = np_xywh_to_xyxy(boxes_flat[bi, mask])
                scores_c = scores[mask]
                classes_c = np.full((len(scores_c),), c, dtype=np.int32)

                all_boxes[bi].append(boxes)
                all_scores[bi].append(scores_c)
                all_classes[bi].append(classes_c)

    results = []

    for bi in range(B):
        if len(all_boxes[bi]) == 0:
            results.append({
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
            })
            continue

        boxes = np.concatenate(all_boxes[bi], axis=0)
        scores = np.concatenate(all_scores[bi], axis=0)
        classes = np.concatenate(all_classes[bi], axis=0)

        final_boxes, final_scores, final_classes = [], [], []

        for c in range(cfg.NUM_CLASSES):
            cmask = classes == c
            if not np.any(cmask):
                continue

            b_c = boxes[cmask]
            s_c = scores[cmask]
            keep = nms_numpy(b_c, s_c, cfg.NMS_IOU, cfg.MAX_DET)

            final_boxes.append(b_c[keep])
            final_scores.append(s_c[keep])
            final_classes.append(np.full((len(keep),), c, dtype=np.int32))

        if len(final_boxes) == 0:
            results.append({
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
            })
        else:
            boxes = np.concatenate(final_boxes, axis=0)
            scores = np.concatenate(final_scores, axis=0)
            classes = np.concatenate(final_classes, axis=0)
            order = scores.argsort()[::-1][:cfg.MAX_DET]

            results.append({
                "boxes": boxes[order].astype(np.float32),
                "scores": scores[order].astype(np.float32),
                "classes": classes[order].astype(np.int32),
            })

    return results


def get_ground_truth(pair, cfg):
    img_path, lbl_path = pair
    labels = load_yolo_labels(lbl_path, cfg.NUM_CLASSES)

    if len(labels) == 0:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int32),
        }

    return {
        "boxes": np_xywh_to_xyxy(labels[:, 1:5]).astype(np.float32),
        "classes": labels[:, 0].astype(np.int32),
    }


def run_inference_on_dataset(model, dataset, loader, cfg):
    model.eval()
    predictions = {}
    ground_truths = {}

    img_counter = 0

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(cfg.AMP and torch.cuda.is_available())):
                outputs = model(images)
            batch_preds = decode_predictions(outputs, cfg)

            for j in range(images.shape[0]):
                pair_idx = batch_idx * loader.batch_size + j
                if pair_idx >= len(dataset.pairs):
                    continue

                predictions[img_counter] = batch_preds[j]
                ground_truths[img_counter] = get_ground_truth(dataset.pairs[pair_idx], cfg)
                img_counter += 1

    return predictions, ground_truths


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def evaluate_predictions(predictions, ground_truths, num_classes, iou_thresholds):
    per_class = {}
    pr_data = {}

    for c in range(num_classes):
        gt_count = sum(int(np.sum(gt["classes"] == c)) for gt in ground_truths.values())

        class_aps = []
        ap50_precision = None
        ap50_recall = None

        for thr in iou_thresholds:
            preds_flat = []

            for img_id, pred in predictions.items():
                mask = pred["classes"] == c
                for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                    preds_flat.append((img_id, float(score), box))

            preds_flat.sort(key=lambda x: x[1], reverse=True)

            matched = {
                img_id: np.zeros(len(ground_truths[img_id]["boxes"]), dtype=bool)
                for img_id in ground_truths.keys()
            }

            tp = np.zeros(len(preds_flat), dtype=np.float32)
            fp = np.zeros(len(preds_flat), dtype=np.float32)

            for i, (img_id, score, box) in enumerate(preds_flat):
                gt = ground_truths[img_id]
                gt_mask = gt["classes"] == c
                gt_boxes = gt["boxes"][gt_mask]
                gt_indices = np.where(gt_mask)[0]

                if len(gt_boxes) == 0:
                    fp[i] = 1.0
                    continue

                ious = iou_numpy(box, gt_boxes)
                best_idx = int(np.argmax(ious))
                best_iou = ious[best_idx]
                original_gt_idx = gt_indices[best_idx]

                if best_iou >= thr and not matched[img_id][original_gt_idx]:
                    tp[i] = 1.0
                    matched[img_id][original_gt_idx] = True
                else:
                    fp[i] = 1.0

            if gt_count == 0:
                ap = np.nan
                precision = np.array([0.0])
                recall = np.array([0.0])
            else:
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recall = tp_cum / (gt_count + 1e-7)
                precision = tp_cum / (tp_cum + fp_cum + 1e-7)
                ap = compute_ap(recall, precision)

            class_aps.append(ap)

            if abs(thr - 0.5) < 1e-9:
                ap50_precision = precision
                ap50_recall = recall
                pr_data[c] = {"precision": precision, "recall": recall}

        valid_aps = [a for a in class_aps if not np.isnan(a)]

        per_class[c] = {
            "AP50": class_aps[0] if len(class_aps) else np.nan,
            "AP75": class_aps[5] if len(class_aps) > 5 else np.nan,
            "AP50_95": float(np.mean(valid_aps)) if valid_aps else np.nan,
            "gt_count": gt_count,
        }

        if ap50_precision is not None and len(ap50_precision):
            f1 = 2 * ap50_precision * ap50_recall / (ap50_precision + ap50_recall + 1e-7)
            best = int(np.argmax(f1))
            per_class[c]["precision"] = float(ap50_precision[best])
            per_class[c]["recall"] = float(ap50_recall[best])
            per_class[c]["F1"] = float(f1[best])
        else:
            per_class[c]["precision"] = 0.0
            per_class[c]["recall"] = 0.0
            per_class[c]["F1"] = 0.0

    ap50_values = [v["AP50"] for v in per_class.values() if not np.isnan(v["AP50"])]
    map_values = [v["AP50_95"] for v in per_class.values() if not np.isnan(v["AP50_95"])]

    overall = {
        "mAP50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "mAP50_95": float(np.mean(map_values)) if map_values else 0.0,
        "macro_precision": float(np.mean([v["precision"] for v in per_class.values()])),
        "macro_recall": float(np.mean([v["recall"] for v in per_class.values()])),
        "macro_F1": float(np.mean([v["F1"] for v in per_class.values()])),
    }

    return {"overall": overall, "per_class": per_class}, pr_data


def save_metrics(metrics, class_names, out_dir, prefix="test"):
    out_dir = Path(out_dir)

    with open(out_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    rows = []
    for c, m in metrics["per_class"].items():
        c = int(c)
        rows.append({
            "class_id": c,
            "class_name": class_names[c] if c < len(class_names) else str(c),
            **m,
        })

    if rows:
        keys = list(rows[0].keys())
        with open(out_dir / f"{prefix}_per_class_metrics.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")

    with open(out_dir / f"{prefix}_overall_metrics.csv", "w") as f:
        keys = list(metrics["overall"].keys())
        f.write(",".join(keys) + "\n")
        f.write(",".join(str(metrics["overall"][k]) for k in keys) + "\n")


def save_pr_curves(pr_data, class_names, out_dir, prefix="test"):
    out_dir = Path(out_dir)

    for c, data in pr_data.items():
        c = int(c)
        plt.figure(figsize=(6, 5))
        plt.plot(data["recall"], data["precision"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        title = class_names[c] if c < len(class_names) else str(c)
        plt.title(f"PR Curve @ IoU 0.50 - {title}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_pr_curve_class_{c}_{title}.png", dpi=300)
        plt.close()


def compute_confusion_matrix(predictions, ground_truths, num_classes, iou_thr=0.5):
    bg = num_classes
    mat = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    for img_id, gt in ground_truths.items():
        pred = predictions[img_id]

        gt_boxes = gt["boxes"]
        gt_classes = gt["classes"]
        pred_boxes = pred["boxes"]
        pred_classes = pred["classes"]
        pred_scores = pred["scores"]

        pred_order = np.argsort(pred_scores)[::-1]
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        for pi in pred_order:
            pbox = pred_boxes[pi]
            pcl = pred_classes[pi]

            if len(gt_boxes) == 0:
                mat[bg, pcl] += 1
                continue

            ious = iou_numpy(pbox, gt_boxes)
            gi = int(np.argmax(ious))

            if ious[gi] >= iou_thr and not matched_gt[gi]:
                mat[gt_classes[gi], pcl] += 1
                matched_gt[gi] = True
            else:
                mat[bg, pcl] += 1

        for gi, matched in enumerate(matched_gt):
            if not matched:
                mat[gt_classes[gi], bg] += 1

    return mat


def save_confusion_matrix(mat, class_names, out_dir, prefix="test"):
    out_dir = Path(out_dir)
    labels = list(class_names) + ["background"]

    plt.figure(figsize=(max(6, len(labels) * 0.75), max(5, len(labels) * 0.65)))
    plt.imshow(mat, interpolation="nearest")
    plt.title("Confusion Matrix @ IoU 0.50")
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")

    thresh = mat.max() / 2.0 if mat.max() > 0 else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(
                j, i, str(mat[i, j]),
                ha="center", va="center",
                color="white" if mat[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close()


def save_predictions_csv(predictions, ground_truths, out_dir, prefix="test"):
    out_dir = Path(out_dir)

    with open(out_dir / f"{prefix}_predictions.csv", "w") as f:
        f.write("image_id,class_id,score,x1,y1,x2,y2\n")
        for img_id, pred in predictions.items():
            for box, score, cls in zip(pred["boxes"], pred["scores"], pred["classes"]):
                f.write(f"{img_id},{int(cls)},{float(score)},{box[0]},{box[1]},{box[2]},{box[3]}\n")

    with open(out_dir / f"{prefix}_ground_truth.csv", "w") as f:
        f.write("image_id,class_id,x1,y1,x2,y2\n")
        for img_id, gt in ground_truths.items():
            for box, cls in zip(gt["boxes"], gt["classes"]):
                f.write(f"{img_id},{int(cls)},{box[0]},{box[1]},{box[2]},{box[3]}\n")


def measure_latency(model, img_size, out_dir, amp=True):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    with torch.inference_mode():
        for _ in range(5):
            with torch.cuda.amp.autocast(enabled=(amp and torch.cuda.is_available())):
                _ = model(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()
        runs = 20
        for _ in range(runs):
            with torch.cuda.amp.autocast(enabled=(amp and torch.cuda.is_available())):
                _ = model(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    sec = (time.time() - t0) / runs
    fps = 1.0 / sec if sec > 0 else 0

    with open(Path(out_dir) / "latency_estimate.json", "w") as f:
        json.dump({
            "seconds_per_image": sec,
            "fps_estimate": fps,
            "batch_size": 1,
            "img_size": img_size,
            "gpu_count": torch.cuda.device_count(),
        }, f, indent=2)

    print(f"Latency: {sec:.4f} sec/image | FPS: {fps:.2f}")


def predict_single_image(model, image_path, cfg):
    image_np = resize_and_load_image(Path(image_path), cfg.IMG_SIZE)
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(cfg.AMP and torch.cuda.is_available())):
            outputs = model(image_t)
        pred = decode_predictions(outputs, cfg)[0]

    return image_np, pred


def plot_prediction(image_np, pred, class_names, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_np)
    H, W = image_np.shape[:2]

    for box, score, cls in zip(pred["boxes"], pred["scores"], pred["classes"]):
        x1, y1, x2, y2 = box
        x1 *= W
        x2 *= W
        y1 *= H
        y2 *= H

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        label = f"{class_names[int(cls)]}: {score:.2f}"
        ax.text(
            x1,
            max(0, y1 - 4),
            label,
            color="white",
            fontsize=9,
            bbox=dict(facecolor="red", alpha=0.7),
        )

    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


