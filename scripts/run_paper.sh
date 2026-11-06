#!/usr/bin/env bash
set -euo pipefail

DATA=${1:-data}
OUT=${2:-results}
BENCHES="mantis mibench mirb muirbench"

run() { python scripts/eval.py --data_root "$DATA" --out_dir "$OUT" --bench $BENCHES "$@"; }

for m in qwen2.5-vl-7b qwen3-vl-8b llava-ov-7b; do
  run --model "$m" --method vanilla
done

for r in 0.453 0.235 0.046; do run --model qwen2.5-vl-7b --ratio "$r"; done

for r in 0.235 0.046; do run --model qwen3-vl-8b --ratio "$r"; done

for b in 5000 3300 1700; do run --model llava-ov-7b --budget "$b"; done

for w in "1 0 0" "0 1 0" "0 0 1" "1 1 0" "1 0 1" "0 1 1"; do
  set -- $w
  run --model qwen2.5-vl-7b --ratio 0.046 --lambda_r "$1" --lambda_c "$2" --lambda_u "$3" --tag "lam$1$2$3"
done

for s in global global_min_quota; do run --model qwen2.5-vl-7b --ratio 0.046 --selection "$s" --tag "$s"; done
for p in 2 4 8 64 128; do run --model qwen2.5-vl-7b --ratio 0.046 --num_prototypes "$p" --tag "P$p"; done
