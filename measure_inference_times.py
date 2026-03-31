from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from time import perf_counter

import torch
from mipcandy import auto_device
from PIL import Image
from sort_screws import (
    ConvNeXtPredictor,
    EfficientNetPredictor,
    ResNetPredictor,
    SwinV2Predictor,
)
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor


MODEL_SPECS: tuple[tuple[str, type, str], ...] = (
    ("EfficientNet", EfficientNetPredictor, "EfficientNetTrainer"),
    ("ResNet", ResNetPredictor, "ResNetTrainer"),
    ("SwinV2", SwinV2Predictor, "SwinV2Trainer"),
    ("ConvNeXt", ConvNeXtPredictor, "ConvNeXtTrainer"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure inference times for all four trained SortScrews models."
    )
    parser.add_argument(
        "--trainer-root",
        type=Path,
        default=Path("trainer"),
        help="Folder containing the four trainer directories.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image to run through each model. Defaults to the first available trainer input.png.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=512,
        help="ROI size used when constructing each predictor.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Final square image size passed to predict_image, matching the live sorter pipeline.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup inferences per model.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of timed inferences per model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Torch device string. Defaults to mipcandy.auto_device().',
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best.pth",
        help="Checkpoint filename inside each trainer final directory.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes expected by the trained checkpoints.",
    )
    return parser.parse_args()


def load_image_tensor(image_path: Path, resize: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = pil_to_tensor(image).float() / 255.0
    return Resize((resize, resize))(tensor)


def resolve_sample_image(trainer_root: Path, explicit_image: Path | None) -> Path:
    if explicit_image is not None:
        if not explicit_image.is_file():
            raise FileNotFoundError(f"Image not found: {explicit_image}")
        return explicit_image
    for _, _, trainer_name in MODEL_SPECS:
        candidate = trainer_root / trainer_name / "final" / "input.png"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No sample image found. Pass --image or place input.png in a trainer final directory."
    )


def sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_model(
    predictor_cls: type,
    experiment_folder: Path,
    checkpoint: str,
    image: torch.Tensor,
    roi_size: int,
    num_classes: int,
    warmup: int,
    repeats: int,
    device: str,
) -> dict[str, float | int]:
    predictor = predictor_cls(
        str(experiment_folder),
        (3, roi_size, roi_size),
        checkpoint=checkpoint,
        device=device,
    )
    predictor.num_classes = num_classes

    with torch.inference_mode():
        for _ in range(warmup):
            predictor.predict_image(image)
        sync_device(device)

        durations_ms: list[float] = []
        for _ in range(repeats):
            start = perf_counter()
            predictor.predict_image(image)
            sync_device(device)
            durations_ms.append((perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.fmean(durations_ms),
        "median_ms": statistics.median(durations_ms),
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
        "std_ms": statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0,
        "fps": 1000.0 / statistics.fmean(durations_ms),
    }


def main() -> None:
    args = parse_args()
    device = args.device or str(auto_device())
    trainer_root: Path = args.trainer_root
    image_path = resolve_sample_image(trainer_root, args.image)
    image = load_image_tensor(image_path, args.resize)

    print(f"device: {device}")
    print(f"image: {image_path}")
    print(f"input shape: {tuple(image.shape)}")
    print(f"warmup: {args.warmup}")
    print(f"repeats: {args.repeats}")
    print()
    print("model           mean (ms)  median (ms)  std (ms)  min (ms)  max (ms)  fps")
    print("--------------  ---------  -----------  --------  --------  --------  -----")

    for model_name, predictor_cls, trainer_name in MODEL_SPECS:
        experiment_folder = trainer_root / trainer_name / "final"
        if not experiment_folder.is_dir():
            print(f"{model_name:<14} missing trainer folder: {experiment_folder}")
            continue
        checkpoint_path = experiment_folder / args.checkpoint
        if not checkpoint_path.is_file():
            print(f"{model_name:<14} missing checkpoint: {checkpoint_path}")
            continue

        try:
            metrics = benchmark_model(
                predictor_cls,
                experiment_folder,
                args.checkpoint,
                image,
                args.roi_size,
                args.num_classes,
                args.warmup,
                args.repeats,
                device,
            )
        except Exception as exc:
            print(f"{model_name:<14} failed: {exc}")
            continue

        print(
            f"{model_name:<14} "
            f"{metrics['mean_ms']:>9.2f}  "
            f"{metrics['median_ms']:>11.2f}  "
            f"{metrics['std_ms']:>8.2f}  "
            f"{metrics['min_ms']:>8.2f}  "
            f"{metrics['max_ms']:>8.2f}  "
            f"{metrics['fps']:>5.2f}"
        )


if __name__ == "__main__":
    main()
