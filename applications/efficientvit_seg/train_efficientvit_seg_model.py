"""Lightweight, controlled DDP trainer for EfficientViT E0--E3 segmentation.

The official EfficientViT release does not include its segmentation trainer.
This file therefore defines a transparent common protocol for hypothesis tests;
it is not presented as the unreleased paper training recipe.
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.seg_model_zoo import create_efficientvit_seg_experiment_model


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
CITYSCAPES_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
CITYSCAPES_TRAIN_IDS = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}
STOP_REQUESTED = False


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; a resumable checkpoint will be saved after this optimizer step.")


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without padding or duplicating samples."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset: str,
        split: str,
        crop_size: tuple[int, int],
        training: bool,
        scale_range: tuple[float, float] = (0.5, 2.0),
        max_samples: int = 0,
    ) -> None:
        self.root = Path(root).expanduser()
        self.dataset = dataset
        self.split = split
        self.crop_size = crop_size
        self.training = training
        self.scale_range = scale_range
        self.samples = self._find_samples()
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise FileNotFoundError(
                f"No {dataset} {split} samples found below {self.root}. "
                "Check the expected directory layout in applications/efficientvit_seg/README.md."
            )

    @property
    def num_classes(self) -> int:
        return 19 if self.dataset == "cityscapes" else 150

    @property
    def class_names(self) -> tuple[str, ...]:
        if self.dataset == "cityscapes":
            return CITYSCAPES_CLASSES
        return tuple(f"class_{index}" for index in range(self.num_classes))

    def _find_samples(self) -> list[tuple[Path, Path]]:
        if self.dataset == "cityscapes":
            image_root = self.root / "leftImg8bit" / self.split
            label_root = self.root / "gtFine" / self.split
            samples = []
            for image_path in sorted(image_root.glob("*/*_leftImg8bit.png")):
                label_name = image_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                label_path = label_root / image_path.parent.name / label_name
                if label_path.is_file():
                    samples.append((image_path, label_path))
            return samples

        split_name = "training" if self.split == "train" else "validation"
        image_root = self.root / "images" / split_name
        label_root = self.root / "annotations" / split_name
        samples = []
        for image_path in sorted(image_root.glob("*.jpg")):
            label_path = label_root / f"{image_path.stem}.png"
            if label_path.is_file():
                samples.append((image_path, label_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _map_labels(self, label: np.ndarray) -> np.ndarray:
        if self.dataset == "cityscapes":
            mapped = np.full(label.shape, -1, dtype=np.int64)
            for raw_id, train_id in CITYSCAPES_TRAIN_IDS.items():
                mapped[label == raw_id] = train_id
            return mapped
        return label.astype(np.int64) - 1

    @staticmethod
    def _pad_to_crop(image: np.ndarray, label: np.ndarray, crop_h: int, crop_w: int):
        height, width = image.shape[:2]
        pad_h = max(crop_h - height, 0)
        pad_w = max(crop_w - width, 0)
        if pad_h == 0 and pad_w == 0:
            return image, label
        mean_color = tuple(float(value * 255) for value in IMAGENET_MEAN)
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=mean_color)
        label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=-1)
        return image, label

    def _train_transform(self, image: np.ndarray, label: np.ndarray):
        crop_h, crop_w = self.crop_size
        scale = random.uniform(*self.scale_range)
        height, width = image.shape[:2]
        scaled_h = max(int(round(height * scale)), 1)
        scaled_w = max(int(round(width * scale)), 1)
        image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        image, label = self._pad_to_crop(image, label, crop_h, crop_w)
        height, width = image.shape[:2]
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        image = image[top : top + crop_h, left : left + crop_w]
        label = label[top : top + crop_h, left : left + crop_w]
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            label = np.ascontiguousarray(label[:, ::-1])
        return image, label

    def _validation_transform(self, image: np.ndarray, label: np.ndarray):
        crop_h, crop_w = self.crop_size
        image = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, label_path = self.samples[index]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        label = self._map_labels(np.asarray(Image.open(label_path)))
        if self.training:
            image, label = self._train_transform(image, label)
        else:
            image, label = self._validation_transform(image, label)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return {"data": image, "label": label, "index": torch.tensor(index)}


def setup_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return local_rank, rank, world_size, device


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def extract_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    for key in ("model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return {key.removeprefix("module."): value for key, value in checkpoint.items() if torch.is_tensor(value)}


def load_compatible_weights(model: torch.nn.Module, checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape
    }
    shape_mismatch = sorted(
        key for key, value in state_dict.items() if key in model_state and model_state[key].shape != value.shape
    )
    incompatible = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "loaded": len(compatible),
        "missing": sorted(incompatible.missing_keys),
        "unexpected": sorted(key for key in state_dict if key not in model_state),
        "shape_mismatch": shape_mismatch,
    }


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )


def learning_rate_at_step(
    step: int,
    max_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def compute_confusion_matrix(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    valid = (target >= 0) & (target < num_classes)
    encoded = target[valid] * num_classes + prediction[valid]
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes).double()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    amp: bool,
    world_size: int,
) -> tuple[float, list[float]]:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    for batch in loader:
        images = batch["data"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(images)
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        predictions = logits.argmax(dim=1)
        confusion += compute_confusion_matrix(predictions, targets, num_classes)
    if world_size > 1:
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
    intersection = confusion.diag()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
    valid_classes = union > 0
    class_iou = torch.full_like(union, float("nan"))
    class_iou[valid_classes] = intersection[valid_classes] / union[valid_classes]
    miou = class_iou[valid_classes].mean().item() * 100
    return miou, (class_iou * 100).cpu().tolist()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    epoch: int,
    best_miou: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_miou": best_miou,
            "args": vars(args),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def save_model_weights(path: Path, model: torch.nn.Module) -> None:
    """Save an evaluator-compatible model artifact without optimizer state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": unwrap_model(model).state_dict()}, temporary_path)
    os.replace(temporary_path, path)


def load_resume_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(extract_state_dict(checkpoint), strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    return int(checkpoint["step"]), int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_miou", -1))


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, allow_nan=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["cityscapes", "ade20k"], default="cityscapes")
    parser.add_argument("--model", default="efficientvit-seg-b1-cityscapes")
    parser.add_argument("--experiment", choices=["e0", "e1", "e2", "e3"], default="e0")
    parser.add_argument("--mhla-num-groups", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-process batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--crop-width", type=int, default=1024)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=6.0e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--val-max-samples", type=int, default=500, help="0 evaluates the full validation split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-bn", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps < 1 or args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("max-steps, batch-size, and grad-accum-steps must be positive")
    if args.workers < 0 or args.crop_height < 1 or args.crop_width < 1:
        raise ValueError("workers must be non-negative and crop dimensions must be positive")
    if args.log_interval < 1 or args.save_interval < 1 or args.eval_interval < 1:
        raise ValueError("log, save, and evaluation intervals must be positive")
    if args.val_max_samples < 0 or args.warmup_steps < 0:
        raise ValueError("val-max-samples and warmup-steps must be non-negative")
    if args.lr <= 0 or not 0 <= args.min_lr_ratio <= 1 or args.weight_decay < 0:
        raise ValueError("Expected lr > 0, 0 <= min-lr-ratio <= 1, and weight-decay >= 0")
    if args.mhla_num_groups < 1:
        raise ValueError("mhla-num-groups must be positive")
    if args.scale_min <= 0 or args.scale_min > args.scale_max:
        raise ValueError("Expected 0 < scale-min <= scale-max")
    expected_dataset = args.model.rsplit("-", 1)[-1]
    if expected_dataset != args.dataset:
        raise ValueError(f"Model {args.model!r} targets {expected_dataset!r}, not {args.dataset!r}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    local_rank, rank, world_size, device = setup_distributed()
    is_master = rank == 0
    seed_everything(args.seed, rank)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    output_dir = Path(args.output_dir).expanduser()
    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    if world_size > 1:
        dist.barrier()

    crop_size = (args.crop_height, args.crop_width)
    train_dataset = SegmentationDataset(
        args.data_root,
        args.dataset,
        "train",
        crop_size,
        training=True,
        scale_range=(args.scale_min, args.scale_max),
    )
    val_dataset = SegmentationDataset(
        args.data_root,
        args.dataset,
        "val",
        crop_size,
        training=False,
        max_samples=args.val_max_samples,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    val_sampler = DistributedEvalSampler(val_dataset, rank, world_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=max(args.workers // 2, 0),
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 1,
    )

    model = create_efficientvit_seg_experiment_model(
        name=args.model,
        experiment=args.experiment,
        mhla_num_groups=args.mhla_num_groups,
        pretrained=False,
    )
    initialization_report = None
    if args.init_checkpoint:
        initialization_report = load_compatible_weights(model, args.init_checkpoint)
    if args.sync_bn and world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    start_step, epoch, best_miou = 0, 0, -1.0
    last_checkpoint = output_dir / "last.pt"
    if args.resume and last_checkpoint.is_file():
        start_step, epoch, best_miou = load_resume_checkpoint(last_checkpoint, model, optimizer, scaler)

    if is_master:
        parameter_count = sum(parameter.numel() for parameter in unwrap_model(model).parameters())
        print(
            f"Training {args.experiment.upper()} on {args.dataset}: world_size={world_size}, "
            f"global_batch={args.batch_size * world_size * args.grad_accum_steps}, params={parameter_count:,}, "
            f"steps={start_step}->{args.max_steps}"
        )
        if initialization_report is not None:
            (output_dir / "initialization_report.json").write_text(
                json.dumps(initialization_report, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"Initialization loaded={initialization_report['loaded']}, "
                f"missing={len(initialization_report['missing'])}, "
                f"shape_mismatch={len(initialization_report['shape_mismatch'])}"
            )

    train_sampler.set_epoch(epoch)
    train_iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output_dir / "metrics.jsonl"
    interval_loss = 0.0
    interval_start = time.perf_counter()

    for step in range(start_step, args.max_steps):
        model.train()
        lr = learning_rate_at_step(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr

        accumulated_loss = 0.0
        for accumulation_index in range(args.grad_accum_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            images = batch["data"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            synchronization_context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and accumulation_index < args.grad_accum_steps - 1
                else nullcontext()
            )
            with synchronization_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    logits = model(images)
                    logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logits, targets, ignore_index=-1)
                    scaled_loss = loss / args.grad_accum_steps
                scaler.scale(scaled_loss).backward()
            accumulated_loss += loss.detach().item()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1

        loss_tensor = torch.tensor(accumulated_loss / args.grad_accum_steps, device=device)
        if world_size > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= world_size
        interval_loss += loss_tensor.item()

        if completed_step % args.log_interval == 0 and is_master:
            elapsed = time.perf_counter() - interval_start
            payload = {
                "type": "train",
                "step": completed_step,
                "epoch": epoch,
                "loss": interval_loss / args.log_interval,
                "lr": lr,
                "seconds_per_step": elapsed / args.log_interval,
            }
            print(json.dumps(payload))
            append_jsonl(metrics_path, payload)
            interval_loss = 0.0
            interval_start = time.perf_counter()

        should_evaluate = completed_step % args.eval_interval == 0 or completed_step == args.max_steps
        if should_evaluate:
            miou, class_iou = evaluate(
                model,
                val_loader,
                device,
                train_dataset.num_classes,
                args.amp,
                world_size,
            )
            if is_master:
                payload = {
                    "type": "validation",
                    "step": completed_step,
                    "epoch": epoch,
                    "miou": miou,
                    "class_iou": dict(zip(val_dataset.class_names, class_iou)),
                }
                print(json.dumps(payload, allow_nan=True))
                append_jsonl(metrics_path, payload)
                if miou > best_miou:
                    best_miou = miou
                    save_model_weights(output_dir / "best.pt", model)

        should_save = completed_step % args.save_interval == 0 or completed_step == args.max_steps
        stop_tensor = torch.tensor(int(STOP_REQUESTED), device=device)
        if world_size > 1:
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
        stop_requested = bool(stop_tensor.item())
        if is_master and (should_save or stop_requested):
            save_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                scaler,
                completed_step,
                epoch,
                best_miou,
                args,
            )

        if stop_requested:
            if world_size > 1:
                dist.barrier()
            break

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
