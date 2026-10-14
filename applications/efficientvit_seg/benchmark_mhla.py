"""Profile the controlled EfficientViT E0--E3 variants."""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.seg_model_zoo import create_efficientvit_seg_experiment_model


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_macs(model: torch.nn.Module, sample: torch.Tensor) -> tuple[int | None, str | None]:
    try:
        from torchprofile import profile_macs as torchprofile_macs

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            macs = int(torchprofile_macs(model, sample))
        note = "; ".join(sorted({str(item.message) for item in caught})) or None
        return macs, note
    except Exception as error:  # profiler coverage varies by PyTorch release/operator
        return None, f"{type(error).__name__}: {error}"


def benchmark_latency(
    model: torch.nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, int | None]:
    device = sample.device
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                model(sample)
            end.record()
            torch.cuda.synchronize(device)
            latency_ms = start.elapsed_time(end) / iterations
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            start_time = time.perf_counter()
            for _ in range(iterations):
                model(sample)
            latency_ms = (time.perf_counter() - start_time) * 1000 / iterations
            peak_memory = None
    return latency_ms, peak_memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientvit-seg-b1-cityscapes")
    parser.add_argument("--variants", nargs="+", choices=["e0", "e1", "e2", "e3"], default=["e0", "e1", "e2", "e3"])
    parser.add_argument("--mhla_num_groups", type=int, default=4)
    parser.add_argument("--input_size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=[512, 1024])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--skip_macs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/mhla_benchmark.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    height, width = args.input_size
    rows = []
    for variant in args.variants:
        model = create_efficientvit_seg_experiment_model(
            name=args.model,
            experiment=variant,
            mhla_num_groups=args.mhla_num_groups,
            pretrained=False,
        ).to(device)
        sample = torch.randn(args.batch_size, 3, height, width, device=device)
        parameters, trainable_parameters = count_parameters(model)
        macs, macs_error = (None, "skipped") if args.skip_macs else profile_macs(model, sample)
        latency_ms, peak_memory = benchmark_latency(model, sample, args.warmup, args.iterations)
        rows.append(
            {
                "experiment": variant.upper(),
                "parameters": parameters,
                "trainable_parameters": trainable_parameters,
                "macs": macs,
                "flops_2_per_mac": None if macs is None else 2 * macs,
                "macs_notes": macs_error,
                "latency_ms": latency_ms,
                "fps": args.batch_size * 1000 / latency_ms,
                "peak_gpu_memory_bytes": peak_memory,
            }
        )
        print(json.dumps(rows[-1], indent=2))

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
        git_dirty = None
    payload = {
        "model": args.model,
        "input_shape": [args.batch_size, 3, height, width],
        "mhla_num_groups": args.mhla_num_groups,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
