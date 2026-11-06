# RI-Prune

```bash
pip install -r requirements.txt
python scripts/download_data.py --data_root data
python scripts/eval.py --model qwen2.5-vl-7b --bench mantis --ratio 0.046
bash scripts/run_paper.sh data results
```
