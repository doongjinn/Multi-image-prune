#!/usr/bin/env bash
# Every setting of the main tables (1-3) and the ablations (Tables 4, 6, 7).
# Usage: bash scripts/run_paper.sh [data_root] [out_dir]
set -euo pipefail

DATA=${1:-data}
OUT=${2:-results}
BENCHES="mantis mibench mirb muirbench"

run() { python scripts/eval.py --data_root "$DATA" --out_dir "$OUT" --bench $BENCHES "$@"; }

# Unpruned baselines: budget-independent, one run per model.
for m in qwen2.5-vl-7b qwen3-vl-8b llava-ov-7b; do
  run --model "$m" --method vanilla
done

# Table 1: Qwen2.5-VL-7B, retained ratios 45.3% / 23.5% / 4.6%
for r in 0.453 0.235 0.046; do run --model qwen2.5-vl-7b --ratio "$r"; done

# Table 3: Qwen3-VL-8B, retained ratios 23.5% / 4.6%
for r in 0.235 0.046; do run --model qwen3-vl-8b --ratio "$r"; done

# Table 2: LLaVA-OneVision-7B, token budgets 5000 / 3300 / 1700
for b in 5000 3300 1700; do run --model llava-ov-7b --budget "$b"; done

# Table 4 (ablation of r / c / u), Qwen2.5-VL at 4.6%
for w in "1 0 0" "0 1 0" "0 0 1" "1 1 0" "1 0 1" "0 1 1"; do
  set -- $w
  run --model qwen2.5-vl-7b --ratio 0.046 --lambda_r "$1" --lambda_c "$2" --lambda_u "$3" --tag "lam$1$2$3"
done

# Table 6 (selection mode) and Table 7 (number of prototypes P)
for s in global global_min_quota; do run --model qwen2.5-vl-7b --ratio 0.046 --selection "$s" --tag "$s"; done
for p in 2 4 8 64 128; do run --model qwen2.5-vl-7b --ratio 0.046 --num_prototypes "$p" --tag "P$p"; done
