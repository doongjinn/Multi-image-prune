# RI-Prune: Relation and Information Aware Visual Token Pruning for Multi-Image Vision-Language Models

**CIKM 2026 (Oral)** · [Paper (arXiv:2611.06047)](https://arxiv.org/abs/2611.06047) · [ACM DL](https://doi.org/10.1145/3799682.3841123)

Dongjin Lee, Eunju Lee, Jinhee Jang, Yoonji Lee, Youngbin Kim (Chung-Ang University)

RI-Prune is a **training-free, attention-free** visual token pruner for multi-image LVLMs. It runs once, on the projector outputs, before the first decoder layer. Each visual token gets three scores:

| Signal | Definition | Intuition |
|---|---|---|
| Query relevance *r* | max<sub>l</sub> cos(v<sub>i</sub>, q<sub>l</sub>) | Is the token related to the question? |
| Cross-image correspondence *c* | max<sub>b≠a, m</sub> cos(v<sub>i</sub>, p<sub>m</sub><sup>(b)</sup>) | Does it match content in the *other* images? |
| Intra-image representativeness *u* | s<sub>i</sub> · log(\|V<sup>(a)</sup>\| / n<sub>m<sub>i</sub></sub><sup>(a)</sup>) | Does it represent a rare region of its own image? |

Here p<sub>m</sub><sup>(a)</sup> are P = 16 prototypes per image: the means of contiguous chunks of that image's tokens. The three scores are z-normalized over the input and summed, S = λ<sub>r</sub>z<sub>r</sub> + λ<sub>c</sub>z<sub>c</sub> + λ<sub>u</sub>z<sub>u</sub> with all λ = 1. The budget is then split evenly across images, and each image keeps its own Top-K tokens.

## Installation

```bash
git clone https://github.com/doongjinn/Multi-image-prune.git && cd Multi-image-prune
pip install -r requirements.txt        # or: pip install -e .
```

Tested with Python 3.10, PyTorch 2.8 (CUDA 12.8), and transformers 4.57.1. Supported backbones:

| Alias | Hugging Face model |
|---|---|
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` |
| `llava-ov-7b` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |

## Quick start

```python
from PIL import Image
from ri_prune import RIPrune
from ri_prune.models import load_model

adapter = load_model("Qwen/Qwen2.5-VL-7B-Instruct")
pruner = RIPrune(adapter)                     # paper defaults

images = [Image.open("a.jpg"), Image.open("b.jpg")]
out = pruner.generate(images, "What changed between the two images?", ratio=0.046)   # or budget=128
print(out.text, out.num_kept_visual_tokens, "/", out.num_visual_tokens)
```

The scorer is plain PyTorch and does not depend on the model. To plug it into another pipeline, use:

```python
from ri_prune import ri_prune
keep = ri_prune(visual_tokens,        # [J, d] projector outputs, image by image
                image_token_counts,   # e.g. [1225, 1225, 900]
                query_embeds,         # [L, d] input embeddings of the text tokens
                budget=128)           # -> sorted indices of the tokens to keep
```

## Evaluation

**1. Download the benchmarks.** MUIRBench, MIBench, MIRB and Mantis-Eval are downloaded from the Hugging Face Hub, and their image archives are extracted:

```bash
python scripts/download_data.py --data_root data
```

**2. Run.**

```bash
# RI-Prune, keep 4.6% of the visual tokens
python scripts/eval.py --model qwen2.5-vl-7b --bench mantis mibench mirb muirbench --ratio 0.046

# fixed token budget (as in the LLaVA-OneVision table)
python scripts/eval.py --model llava-ov-7b --bench muirbench --budget 1700

# unpruned model; it does not depend on the budget, so run it once per model
python scripts/eval.py --model qwen2.5-vl-7b --bench muirbench --method vanilla
```

Each run writes `results/<tag>.jsonl` with one line per sample, and `results/<tag>.summary.json` with accuracy, the retained-token ratio and latency. Interrupted runs resume from the `.jsonl` file.

`bash scripts/run_paper.sh data results` runs every setting of the main tables and ablations (Tables 1–4, 6 and 7).

| Option | Default | |
|---|---|---|
| `--ratio` / `--budget` | – | retained fraction of visual tokens, or an absolute count |
| `--num_prototypes` | 16 | P, prototypes per image |
| `--lambda_r --lambda_c --lambda_u` | 1 1 1 | signal weights (0 disables a signal) |
| `--selection` | `per_image` | `per_image`, `global`, `global_min_quota` |
| `--query_scope` | `text` | query tokens for *r*: all text tokens, or only the question (`prompt`) |
| `--n_samples` / `--seed` | 1000 / 42 | seeded random subset per benchmark (`--n_samples 0` uses all samples) |
| `--attn` | `sdpa` | attention implementation (`flash_attention_2` if installed) |

## Implementation notes

- **Where pruning happens.** `prepare()` builds the full input embeddings, with projector outputs written into the image slots. RI-Prune selects tokens from those embeddings, and the dropped tokens are removed before the first decoder layer. The vision encoder and projector always process every image, so only the decoder sees fewer tokens.
- **Positions.** Qwen2.5-VL and Qwen3-VL keep the original M-RoPE (t, h, w) position of every surviving token, and generated tokens continue at the original maximum position + 1, as in unpruned decoding. LLaVA-OneVision uses 1-D RoPE and is re-indexed contiguously after pruning.
- **Qwen3-VL deepstack.** Qwen3-VL adds intermediate ViT features to the first decoder layers at the visual positions. These features are pruned with the same indices, so each kept token keeps its deepstack features.
- **Decoding** is greedy and runs its own loop with a KV cache (`ri_prune/models/base.py`). With no pruning, it reproduces plain greedy `model.generate(do_sample=False, repetition_penalty=1.0)` token for token (checked by `scripts/sanity_check.py`). Logits processors from a checkpoint's `generation_config` are not applied: Qwen2.5-VL, for example, ships `repetition_penalty=1.05`, which a bare `model.generate(do_sample=False)` would still use.

```
ri_prune/
  core.py           scoring (r, c, u), z-normalization, per-image Top-K
  pipeline.py       RIPrune: prepare -> select -> generate
  benchmarks.py     MUIRBench, MIBench, MIRB, Mantis-Eval loaders, prompts and scoring
  models/           adapters for Qwen2.5-VL / Qwen3-VL and LLaVA-OneVision
scripts/            eval.py, download_data.py, run_paper.sh, sanity_check.py (GPU checks)
tests/test_core.py  CPU tests against a literal implementation of the equations
```

Run the tests with `python tests/test_core.py` (or `pytest tests/`).

## Citation

```bibtex
@inproceedings{lee2026riprune,
  title     = {RI-Prune: Relation and Information Aware Visual Token Pruning for Multi-Image Vision-Language Models},
  author    = {Lee, Dongjin and Lee, Eunju and Jang, Jinhee and Lee, Yoonji and Kim, Youngbin},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  series    = {CIKM '26},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3799682.3841123}
}
```

## Acknowledgements

This work was supported by the IITP grant funded by the Korea government (MSIT) [RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University)] and by the NRF grant funded by the Korea government (MSIT) (RS-2025-00556246).
