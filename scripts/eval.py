#!/usr/bin/env python
"""Evaluate RI-Prune (or the unpruned model) on multi-image benchmarks.

Examples:
    # RI-Prune keeping 4.6% of visual tokens on MUIRBench with Qwen2.5-VL-7B
    python scripts/eval.py --model qwen2.5-vl-7b --bench muirbench --ratio 0.046

    # fixed token budget on LLaVA-OneVision
    python scripts/eval.py --model llava-ov-7b --bench mantis mibench mirb muirbench --budget 1700

    # unpruned baseline (run once per benchmark; it does not depend on the budget)
    python scripts/eval.py --model qwen2.5-vl-7b --bench muirbench --method vanilla

Outputs (in --out_dir):
    <tag>.jsonl         one line per sample (prediction, score, token counts, latencies)
    <tag>.summary.json  accuracy and averages
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ri_prune import RIPrune, RIPruneConfig  # noqa: E402
from ri_prune.benchmarks import BENCHMARKS, get_benchmark  # noqa: E402
from ri_prune.core import SELECTION_MODES  # noqa: E402
from ri_prune.models import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or alias: qwen2.5-vl-7b, qwen3-vl-8b, llava-ov-7b")
    p.add_argument("--bench", nargs="+", required=True, choices=list(BENCHMARKS))
    p.add_argument("--data_root", default="data", help="directory prepared by scripts/download_data.py")
    p.add_argument("--method", default="ri_prune", choices=["ri_prune", "vanilla"])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--budget", type=int, help="number of visual tokens to keep per sample")
    g.add_argument("--ratio", type=float, help="fraction of visual tokens to keep per sample, in (0, 1]")
    # RI-Prune hyperparameters (paper defaults)
    p.add_argument("--num_prototypes", type=int, default=16, help="P, prototypes per image")
    p.add_argument("--lambda_r", type=float, default=1.0)
    p.add_argument("--lambda_c", type=float, default=1.0)
    p.add_argument("--lambda_u", type=float, default=1.0)
    p.add_argument("--selection", default="per_image", choices=list(SELECTION_MODES))
    p.add_argument("--query_scope", default="text", choices=["text", "prompt"],
                   help="query tokens for r: all text tokens (paper) or only the question")
    # evaluation protocol
    p.add_argument("--n_samples", type=int, default=1000,
                   help="seeded random subset per benchmark (<= 0 for all samples)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--cache_dir", default=None, help="Hugging Face cache directory")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--tag", default="", help="suffix for output file names")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples (smoke test)")
    args = p.parse_args()
    if args.method == "ri_prune" and args.budget is None and args.ratio is None:
        p.error("--method ri_prune needs --budget or --ratio")
    return args


def run_tag(args: argparse.Namespace, model_name: str, bench: str) -> str:
    if args.method == "vanilla":
        tag = f"{model_name}_{bench}_vanilla"
    else:
        size = f"b{args.budget}" if args.budget is not None else f"r{args.ratio:g}"
        tag = f"{model_name}_{bench}_riprune_{size}"
    return tag + (f"_{args.tag}" if args.tag else "")


def evaluate(pruner: RIPrune, bench, args, out_dir: Path, tag: str) -> dict:
    samples = bench.samples()
    if args.limit is not None:
        samples = samples[: args.limit]
    jsonl = out_dir / f"{tag}.jsonl"

    done: dict[str, dict] = {}
    if jsonl.exists():                                     # resume
        for line in jsonl.read_text().splitlines():
            rec = json.loads(line)
            done[rec["idx"]] = rec
    print(f"[{tag}] {len(samples)} samples ({len(done)} already done)")

    t_start = time.time()
    n_new = 0
    with jsonl.open("a") as f:
        for i, s in enumerate(samples):
            if s.idx in done:
                continue
            n_new += 1
            rec = {"idx": s.idx, "num_images": len(s.images), "gold": s.gold}
            try:
                out = pruner.generate(
                    s.images, bench.format_prompt(s),
                    budget=args.budget, ratio=args.ratio,
                    max_new_tokens=args.max_new_tokens, prune=args.method == "ri_prune",
                )
                pred = bench.parse_prediction(s, out.text)
                rec.update(out.to_dict(), prediction=pred, score=bench.score(s, pred))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec.update(error="cuda_oom", score=None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[s.idx] = rec
            if (i + 1) % 25 == 0:
                scored = [r["score"] for r in done.values() if r.get("score") is not None]
                print(f"  {i + 1}/{len(samples)}  acc={sum(scored) / max(len(scored), 1):.4f}  "
                      f"{(time.time() - t_start) / n_new:.2f}s/sample", flush=True)

    recs = [done[s.idx] for s in samples if s.idx in done]
    ok = [r for r in recs if r.get("score") is not None]
    n = max(len(ok), 1)
    mean = lambda key: sum(r[key] for r in ok) / n  # noqa: E731
    summary = {
        "tag": tag,
        "model": args.model,
        "benchmark": bench.name,
        "method": args.method,
        "budget": args.budget,
        "ratio": args.ratio,
        "config": vars(pruner.config),
        "n_samples": len(recs),
        "n_errors": len(recs) - len(ok),
        "accuracy": round(100 * sum(r["score"] for r in ok) / n, 2),
        "avg_visual_tokens": round(mean("num_visual_tokens"), 1) if ok else None,
        "avg_kept_visual_tokens": round(mean("num_kept_visual_tokens"), 1) if ok else None,
        "retained_ratio": (round(sum(r["num_kept_visual_tokens"] for r in ok)
                                 / max(sum(r["num_visual_tokens"] for r in ok), 1), 4) if ok else None),
        "avg_latency_select_sec": round(mean("latency_select_sec"), 4) if ok else None,
        "avg_latency_generate_sec": round(mean("latency_generate_sec"), 4) if ok else None,
        "avg_latency_total_sec": (round(mean("latency_prepare_sec") + mean("latency_select_sec")
                                        + mean("latency_generate_sec"), 4) if ok else None),
    }
    (out_dir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{tag}] accuracy={summary['accuracy']:.2f}  retained={summary['retained_ratio']}  "
          f"errors={summary['n_errors']}")
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = load_model(args.model, attn_implementation=args.attn,
                         cache_dir=args.cache_dir, query_scope=args.query_scope)
    config = RIPruneConfig(
        num_prototypes=args.num_prototypes, lambda_r=args.lambda_r, lambda_c=args.lambda_c,
        lambda_u=args.lambda_u, selection=args.selection,
    )
    pruner = RIPrune(adapter, config)
    model_name = args.model.rstrip("/").split("/")[-1]

    for name in args.bench:
        bench = get_benchmark(name, args.data_root,
                              n_samples=args.n_samples if args.n_samples > 0 else None, seed=args.seed)
        evaluate(pruner, bench, args, out_dir, run_tag(args, model_name, name))


if __name__ == "__main__":
    main()
