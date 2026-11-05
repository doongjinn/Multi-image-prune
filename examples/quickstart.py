"""Minimal RI-Prune example: answer a question about two images with 10% of the visual tokens.

    python examples/quickstart.py img1.jpg img2.jpg "What is different between the two images?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from ri_prune import RIPrune, RIPruneConfig
from ri_prune.models import load_model

if len(sys.argv) < 3:
    sys.exit(__doc__)
*paths, question = sys.argv[1:]

adapter = load_model("Qwen/Qwen2.5-VL-7B-Instruct")          # or Qwen/Qwen3-VL-8B-Instruct, llava-hf/llava-onevision-qwen2-7b-ov-hf
pruner = RIPrune(adapter, RIPruneConfig())                     # P=16, lambda_r = lambda_c = lambda_u = 1, per-image Top-K
images = [Image.open(p).convert("RGB") for p in paths]

full = pruner.generate(images, question, prune=False)
pruned = pruner.generate(images, question, ratio=0.10)

print(f"full   ({full.num_visual_tokens} visual tokens): {full.text}")
print(f"pruned ({pruned.num_kept_visual_tokens} visual tokens): {pruned.text}")
