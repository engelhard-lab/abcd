from pathlib import Path
from re import search
from numpy import argmin, argmax


def cleanup_checkpoints(checkpoint_dir, mode="min"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    metrics = [
        (float(match.group(1)), checkpoint)
        for checkpoint in checkpoints
        if (match := search(r"(\d+\.\d+)(?=[^\d]|$)", checkpoint.name))
    ]
    metrics.sort(key=lambda x: x[0], reverse=(mode == "max"))
    for _, checkpoint in metrics[1:]:
        checkpoint.unlink()


def get_best_checkpoint(ckpt_folder: Path, mode: str) -> Path:
    checkpoint_paths = list(ckpt_folder.glob("*"))
    metrics = [
        float(match.group(1))
        for filepath in checkpoint_paths
        if (match := search(r"(\d+\.\d+)(?=[^\d]|$)", filepath.stem))
    ]
    if mode == "min":
        index = argmin(metrics)
    else:
        index = argmax(metrics)
    return checkpoint_paths[index]
