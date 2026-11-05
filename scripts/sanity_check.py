"""GPU sanity checks for RI-Prune.

    python scripts/sanity_check.py --model Qwen/Qwen2.5-VL-7B-Instruct --n 6 \
        [--mantis <Mantis-Eval dir>] [--cache_dir <hf cache>] [--legacy_root <NIPS_HALLU dir>]

1. Unpruned path through our greedy loop == plain greedy HF model.generate (decoded text), on the
   benchmark prompts and on an open-ended prompt with long answers.
2. Pruned runs at the paper ratios produce answers and keep exactly round(ratio*J) tokens.
3. (optional, Qwen2.5-VL) keep set vs the original experiment code (NIPS_HALLU/methods/rig_prune.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ri_prune import RIPrune
from ri_prune.benchmarks import MantisEval
from ri_prune.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--mantis", default=None, help="Mantis-Eval dir (contains mantis_eval/test-*.parquet); downloaded if omitted")
ap.add_argument("--cache_dir", default=None)
ap.add_argument("--legacy_root", default=None)
ap.add_argument("--open_tokens", type=int, default=128, help="max new tokens for the open-ended check")
args = ap.parse_args()

if args.mantis is None:
    from huggingface_hub import snapshot_download
    args.mantis = snapshot_download("TIGER-Lab/Mantis-Eval", repo_type="dataset")

bench = MantisEval(args.mantis, n_samples=None)
samples = [s for s in bench.samples() if len(s.images) >= 2][: args.n]
print("samples", len(samples), [len(s.images) for s in samples])

adapter = load_model(args.model, cache_dir=args.cache_dir)
print(type(adapter).__name__, adapter.model.config.model_type, "eos", adapter.eos_token_ids)
pruner = RIPrune(adapter)
proc, model = adapter.processor, adapter.model

# ---- 1. vanilla equivalence: benchmark prompts (short answers) and an open-ended prompt
# (long answers, so the decode positions and the KV cache are exercised over many steps)
OPEN_PROMPT = "Describe each image in detail and then explain how the images differ."
n_match = 0
for kind, max_new in (("bench", 24), ("open", args.open_tokens)):
    for s in samples:
        text = bench.format_prompt(s) if kind == "bench" else OPEN_PROMPT
        ours = pruner.generate(s.images, text, prune=False, max_new_tokens=max_new)
        prompt = adapter.build_prompt(s.images, text)
        inputs = proc(text=[prompt], images=s.images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # plain greedy: checkpoints ship sampling / repetition_penalty defaults
            # (Qwen2.5-VL: repetition_penalty=1.05) that model.generate would otherwise apply
            g = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0,
                               temperature=None, top_p=None, top_k=None)
        hf = proc.batch_decode(g[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        same = hf == ours.text
        n_match += same
        shown = (lambda t: t if len(t) <= 60 else t[:60] + "...") if same else (lambda t: t)
        print(f"[vanilla/{kind}] J={ours.num_visual_tokens} counts={ours.image_token_counts} "
              f"gen={ours.num_generated_tokens} same={same} ours={shown(ours.text)!r} hf={shown(hf)!r}")
print(f"vanilla match {n_match}/{2 * len(samples)}")

# ---- 2. pruned
for ratio in (0.453, 0.235, 0.046):
    accs = []
    for s in samples:
        text = bench.format_prompt(s)
        out = pruner.generate(s.images, text, ratio=ratio, max_new_tokens=24)
        accs.append(bench.score(s, bench.parse_prediction(s, out.text)))
        assert out.num_kept_visual_tokens == max(1, round(ratio * out.num_visual_tokens)), out
    print(f"[ratio {ratio}] acc={sum(accs)/len(accs):.3f} last: kept {out.num_kept_visual_tokens}/{out.num_visual_tokens} "
          f"sel={out.latency_select_sec*1e3:.1f}ms gen={out.latency_generate_sec:.2f}s text={out.text!r}")
    out = pruner.generate(samples[0].images, OPEN_PROMPT, ratio=ratio, max_new_tokens=48)
    print(f"[ratio {ratio}] open-ended: {out.text!r}")

# ---- 3. keep set vs legacy implementation (Qwen2.5-VL only, needs the old repo)
if args.legacy_root and model.config.model_type == "qwen2_5_vl":
    sys.path.insert(0, args.legacy_root)
    from methods.rig_prune import compute_rig_prune_scores, select_keep_indices_per_image

    for s in samples[:4]:
        prep = adapter.prepare(s.images, bench.format_prompt(s))
        B = round(0.046 * prep.num_visual_tokens)
        new_keep = set(pruner.select(prep, B).tolist())
        offs = [0]
        for c in prep.image_token_counts:
            offs.append(offs[-1] + c)
        dev = prep.visual_tokens.device
        per_img = [torch.arange(offs[a], offs[a + 1], device=dev) for a in range(len(prep.image_token_counts))]
        q_emb = model.get_input_embeddings()(prep.input_ids[0][prep.query_positions])
        legacy = compute_rig_prune_scores(prep.visual_tokens, per_img, q_emb, K=16, lam_r=1.0, mu=1.0, nu=1.0)
        old_keep = set(select_keep_indices_per_image(legacy["score"], per_img, B))
        print(f"[legacy] B={B} overlap={len(new_keep & old_keep)}/{B}")
print("peak mem GB", torch.cuda.max_memory_allocated() / 1e9)
