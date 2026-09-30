"""Verified public example assets, cached outside the source tree."""
import hashlib
from pathlib import Path

import torch

RELEASE = "https://github.com/HUST-ISMD-Odyssey/MemIntelli/releases/download/v2.0.0"
ASSETS = {
    "gru_speech_commands.pt": "fb3e4fad54e615e922f4f640d4747db57f9598b62efb498a076a15203658431e",
    "yolov3_voc.pt": "2a96b402f3818d9c128c98b4abf051f77bbff08bbf2bb632fd616b34dc46f0cc",
    "yolov5s_voc.pt": "9503b0ca95fa365c36725ecd6e2f3bc4b1df84fc26dc114847c9f79c9196ebda",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(name, cache=None):
    if name not in ASSETS:
        raise ValueError(f"Unknown public checkpoint: {name}")
    folder = Path(cache or Path(torch.hub.get_dir()) / "checkpoints" / "memintelli-v2")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if not path.exists():
        partial = path.with_suffix(".download")
        torch.hub.download_url_to_file(f"{RELEASE}/{name}", str(partial), hash_prefix=ASSETS[name])
        partial.replace(path)
    if sha256(path) != ASSETS[name]:
        raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
    return path
