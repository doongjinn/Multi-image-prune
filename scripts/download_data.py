#!/usr/bin/env python
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = {
    "muirbench": ("MUIRBENCH/MUIRBENCH", ["data/*.parquet"]),
    "mibench": ("StarBottle/MIBench", ["*.json", "image.zip"]),
    "mirb": ("VLLMs/MIRB", ["*.json", "images.zip"]),
    "mantis": ("TIGER-Lab/Mantis-Eval", ["mantis_eval/*.parquet"]),
}


def extract(zip_path: Path, dest: Path) -> None:
    marker = dest / f".{zip_path.name}.extracted"
    if marker.exists():
        return
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        root = dest.resolve()
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe path in {zip_path.name}: {member.filename}")
        zf.extractall(dest)
    marker.touch()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--bench", nargs="+", default=list(REPOS), choices=list(REPOS))
    args = p.parse_args()

    for name in args.bench:
        repo, patterns = REPOS[name]
        dest = Path(args.data_root) / name
        print(f"[{name}] {repo} -> {dest}")
        snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=patterns)
        for z in dest.glob("*.zip"):
            extract(z, dest)
    print("done")


if __name__ == "__main__":
    main()
