# RI-Prune

Training-free visual token pruning for multi-image LVLMs. [Paper](https://arxiv.org/abs/2611.06047)

## Setup

```bash
git clone https://github.com/doongjinn/Multi-image-prune.git && cd Multi-image-prune
pip install -r requirements.txt
```

Python 3.10, PyTorch 2.8, transformers 4.57.1.
Models: `qwen2.5-vl-7b`, `qwen3-vl-8b`, `llava-ov-7b`.

## Usage

```python
from PIL import Image
from ri_prune import RIPrune
from ri_prune.models import load_model

pruner = RIPrune(load_model("Qwen/Qwen2.5-VL-7B-Instruct"))
out = pruner.generate([Image.open("a.jpg"), Image.open("b.jpg")], "What changed?", ratio=0.046)  # or budget=1700
print(out.text)
```

## Evaluation

```bash
python scripts/download_data.py --data_root data          # MUIRBench, MIBench, MIRB, Mantis-Eval
python scripts/eval.py --model qwen2.5-vl-7b --bench mantis mibench mirb muirbench --ratio 0.046
python scripts/eval.py --model llava-ov-7b --bench muirbench --budget 1700
python scripts/eval.py --model qwen2.5-vl-7b --bench muirbench --method vanilla   # no pruning
bash scripts/run_paper.sh data results                     # all paper settings
```

Results go to `results/<tag>.jsonl` and `results/<tag>.summary.json`. See `python scripts/eval.py -h` for options.

Tests: `python tests/test_core.py`
