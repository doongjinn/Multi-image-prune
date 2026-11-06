from __future__ import annotations

import io
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

LETTER_RE = re.compile(r"\b([A-Z])\b")


@dataclass
class Sample:
    idx: str
    images: list[Image.Image]
    question: str
    options: list[str] = field(default_factory=list)
    gold: str = ""
    answer_format: str = "lettered"
    meta: dict[str, Any] = field(default_factory=dict)


def _open_image(obj: Any, root: Path | None = None) -> Image.Image | None:
    try:
        if isinstance(obj, dict):
            if obj.get("bytes"):
                return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
            if obj.get("path") and root is not None:
                return Image.open(root / obj["path"]).convert("RGB")
            return None
        if isinstance(obj, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(obj))).convert("RGB")
        if isinstance(obj, str) and root is not None:
            return Image.open(root / obj).convert("RGB")
        if hasattr(obj, "convert"):
            return obj.convert("RGB")
    except (OSError, ValueError):
        return None
    return None


def _read_parquet_rows(files: list[Path]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for f in files:
        for batch in pq.read_table(f).to_batches(max_chunksize=64):
            yield from batch.to_pylist()


def _label_images(text: str, counter: list[int]) -> str:
    parts = text.split("<image>")
    out = parts[0]
    for part in parts[1:]:
        counter[0] += 1
        out += f"[Image {counter[0]}]" + part
    return out


def _mc_prompt(question: str, options: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    lines = [f"({lt}) {opt}" for lt, opt in zip(letters, options)]
    return (f"{question.strip()}\nOptions:\n" + "\n".join(lines) + "\n"
            + f"Answer with exactly one option letter: {', '.join(letters)}.")


def _parse_letter(text: str, n_options: int) -> str | None:
    valid = {chr(ord("A") + i) for i in range(max(n_options, 1))}
    m = LETTER_RE.search(text.strip())
    return m.group(1).upper() if m and m.group(1).upper() in valid else None


def _loose_match(pred: str | None, gold: str) -> float:
    if pred is None or not gold:
        return 0.0
    pred = pred.strip().lower()
    golds = [g.strip().lower() for g in gold.split("|") if g.strip()]
    if not golds:
        return 0.0
    yn = re.search(r"\b(yes|no)\b", pred)
    if yn:
        for g in golds:
            if g in ("yes", "no"):
                return 1.0 if yn.group(1) == g else 0.0
    return 1.0 if any(g in pred for g in golds) else 0.0


class Benchmark:
    name = "base"

    def __init__(self, root: str | Path, n_samples: int | None = None, seed: int = 42) -> None:
        self.root = Path(root)
        self.n_samples = n_samples
        self.seed = seed
        if not self.root.exists():
            raise FileNotFoundError(self.root)

    def load_samples(self) -> list[Sample]:
        raise NotImplementedError

    def format_prompt(self, sample: Sample) -> str:
        raise NotImplementedError

    def parse_prediction(self, sample: Sample, text: str) -> str | None:
        return _parse_letter(text, len(sample.options)) if sample.answer_format == "lettered" else text.strip()

    def score(self, sample: Sample, prediction: str | None) -> float:
        if sample.answer_format == "lettered":
            return float(prediction is not None and prediction.upper() == sample.gold.strip().upper())
        return _loose_match(prediction, sample.gold)

    def samples(self) -> list[Sample]:
        items = self.load_samples()
        if self.n_samples is None or self.n_samples >= len(items):
            return items
        return random.Random(self.seed).sample(items, self.n_samples)


class MUIRBench(Benchmark):
    name = "muirbench"

    def load_samples(self) -> list[Sample]:
        files = sorted(self.root.glob("**/*test*.parquet")) or sorted(self.root.glob("**/*.parquet"))
        out = []
        for row in _read_parquet_rows(files):
            images = [im for im in (_open_image(x) for x in row.get("image_list") or []) if im is not None]
            if images:
                out.append(Sample(idx=str(row["idx"]), images=images, question=row["question"],
                                  options=list(row["options"]), gold=str(row["answer"]),
                                  meta={"task": row.get("task", "")}))
        return out

    def format_prompt(self, sample: Sample) -> str:
        counter = [0]
        question = _label_images(sample.question, counter)
        return _mc_prompt(question, [_label_images(o, counter) for o in sample.options])


class MIBench(Benchmark):
    name = "mibench"
    files = ("multi_image_instruction.json", "multimodal_knowledge_seeking.json")

    def load_samples(self) -> list[Sample]:
        out = []
        for fname in self.files:
            path = self.root / fname
            if not path.exists():
                continue
            for row in json.loads(path.read_text()):
                if row.get("type") != "multiple-choice":
                    continue
                images = [_open_image(rel, self.root) for rel in row.get("image") or []]
                if len(images) < 2 or any(im is None for im in images):
                    continue
                options = [str(o) for o in row.get("options") or []]
                answer = str(row.get("answer", "")).strip().lower()
                gold = next((chr(ord("A") + i) for i, o in enumerate(options) if o.strip().lower() == answer), None)
                if options and gold is not None:
                    out.append(Sample(idx=str(row.get("id", f"mibench_{len(out)}")), images=images,
                                      question=row.get("question", ""), options=options, gold=gold,
                                      meta={"task": row.get("task", "")}))
        return out

    def format_prompt(self, sample: Sample) -> str:
        return _mc_prompt(_label_images(sample.question, [0]), sample.options)


class MIRB(Benchmark):
    name = "mirb"
    tasks = ("analogy", "attribute", "count", "food", "image_jigsaw", "sightseeing", "visual_chain",
             "plot_code", "arxiv", "3d_scene", "count_concat", "visual_chain_concat", "3d_scene_concat", "codeu")

    def load_samples(self) -> list[Sample]:
        out = []
        for task in self.tasks:
            path = self.root / f"{task}.json"
            if not path.exists():
                continue
            for i, row in enumerate(json.loads(path.read_text())):
                images = [_open_image(rel, self.root) for rel in row.get("images") or []]
                if not images or any(im is None for im in images):
                    continue
                gold = row.get("answers", "")
                if isinstance(gold, list):
                    gold = " | ".join(str(g) for g in gold)
                out.append(Sample(idx=f"{task}_{i}", images=images, question=row.get("questions", ""),
                                  gold=str(gold), answer_format="open", meta={"task": task}))
        return out

    def format_prompt(self, sample: Sample) -> str:
        return f"{sample.question.strip()}\nAnswer concisely."


class MantisEval(Benchmark):
    name = "mantis"

    def load_samples(self) -> list[Sample]:
        out = []
        for row in _read_parquet_rows(sorted(self.root.glob("**/*.parquet"))):
            images = [im for im in (_open_image(x) for x in row.get("images") or []) if im is not None]
            if not images:
                continue
            qtype = row.get("question_type", "multi-choice")
            raw = row.get("options") or []
            if isinstance(raw, str):
                raw = [p for p in re.split(r"\([A-Z]\)\s*", raw) if p.strip()]
            options = []
            for o in raw:
                m = re.match(r"^\s*\(?([A-Z])\)?\s*(.*)$", str(o))
                options.append(m.group(2).strip() if m else str(o))
            mc = qtype == "multi-choice" and bool(options)
            out.append(Sample(idx=str(row.get("id", "")), images=images, question=str(row.get("question", "")),
                              options=options if mc else [], gold=str(row.get("answer", "")),
                              answer_format="lettered" if mc else "open", meta={"category": row.get("category", "")}))
        return out

    def format_prompt(self, sample: Sample) -> str:
        question = _label_images(sample.question, [0])
        if sample.answer_format == "lettered":
            return _mc_prompt(question, sample.options)
        return f"{question.strip()}\nAnswer concisely."


BENCHMARKS = {b.name: b for b in (MUIRBench, MIBench, MIRB, MantisEval)}


def get_benchmark(name: str, data_root: str | Path, **kwargs: Any) -> Benchmark:
    return BENCHMARKS[name.lower()](Path(data_root) / name.lower(), **kwargs)
