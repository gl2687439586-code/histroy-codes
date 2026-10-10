from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .common import (
    pick_answer,
    pick_id,
    pick_question,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)


DEFAULT_GOLD_CANDIDATES = [
    "/home/researcher/smartsearch_rebuild/data/hotpotqa/prepared/random500_seed2026/gold_labels.sealed.jsonl",
    "/home/researcher/smartsearch_rebuild/data/hotpotqa/prepared/gold_labels.sealed.jsonl",
    "/home/researcher/smartsearch_rebuild/data/hotpotqa/hotpot_dev_distractor_v1.json",
    "/home/researcher/smartsearch_rebuild/data/hotpotqa/hotpot_dev_fullwiki_v1.json",
]


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("data", "questions", "records", "examples"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
        raise ValueError(f"Unsupported gold-label structure in {path}")
    return value


def resolve_gold(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for candidate in DEFAULT_GOLD_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not locate HotpotQA gold labels. Pass --gold explicitly. Tried:\n"
        + "\n".join(DEFAULT_GOLD_CANDIDATES)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--gold")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected", type=int, default=500)
    args = parser.parse_args()

    query_path = Path(args.queries).resolve()
    if not query_path.is_file():
        raise FileNotFoundError(query_path)
    gold_path = resolve_gold(args.gold).resolve()
    output_dir = Path(args.output_dir).resolve()

    query_rows = read_jsonl(query_path)
    if len(query_rows) != args.expected:
        raise RuntimeError(
            f"Expected {args.expected} query rows, found {len(query_rows)} in {query_path}"
        )

    questions: List[Dict[str, Any]] = []
    ids = set()
    for source_index, row in enumerate(query_rows):
        qid = pick_id(row)
        if qid in ids:
            raise RuntimeError(f"Duplicate query id: {qid}")
        ids.add(qid)
        questions.append(
            {
                "source_index": source_index,
                "sample_id": qid,
                "question": pick_question(row),
                "dataset_name": "hotpotqa",
            }
        )

    gold_by_id: Dict[str, Any] = {}
    for row in load_records(gold_path):
        try:
            qid = pick_id(row)
            answer = pick_answer(row)
        except KeyError:
            continue
        if qid in ids:
            gold_by_id[qid] = answer
    missing = sorted(ids - set(gold_by_id))
    if missing:
        raise RuntimeError(
            f"Gold source is missing {len(missing)} selected ids; first 20: {missing[:20]}"
        )

    labels = [
        {"sample_id": row["sample_id"], "answer": gold_by_id[row["sample_id"]]}
        for row in questions
    ]
    questions_out = output_dir / "questions.no_labels.jsonl"
    labels_out = output_dir / "gold_labels.sealed.jsonl"
    write_jsonl(questions_out, questions)
    write_jsonl(labels_out, labels)
    manifest = {
        "created_at": utc_now(),
        "benchmark": "HotpotQA",
        "records": len(questions),
        "unique_ids": len(ids),
        "contains_gold_answers_in_generation_file": False,
        "queries_source": str(query_path),
        "queries_source_sha256": sha256_file(query_path),
        "gold_source": str(gold_path),
        "gold_source_sha256": sha256_file(gold_path),
        "questions_output": str(questions_out),
        "gold_labels_output": str(labels_out),
    }
    write_json(output_dir / "prepare_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

