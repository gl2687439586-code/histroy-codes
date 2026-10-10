#!/usr/bin/env python3
"""Build the 125-item JSON input used by the Qwen3-only v6 runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def read_json_list(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("data", "items", "samples", "records"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON list")
    return [row for row in value if isinstance(row, dict)]


def first(row: Dict[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, "", []):
            return value
    return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paper_intermediate", default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl).resolve()
    output_path = Path(args.output).resolve()
    dataset = read_jsonl(dataset_path)
    if len(dataset) != 125:
        raise RuntimeError(f"Bamboogle must contain 125 rows; found {len(dataset)}")

    paper_path = Path(args.paper_intermediate).resolve() if args.paper_intermediate else None
    paper_by_id: Dict[str, Dict[str, Any]] = {}
    if paper_path:
        if not paper_path.is_file():
            raise FileNotFoundError(paper_path)
        for index, row in enumerate(read_json_list(paper_path)):
            sample_id = str(first(row, ("id", "sample_id", "qid"), f"test_{index}"))
            paper_by_id[sample_id] = row

    output: List[Dict[str, Any]] = []
    ids = set()
    paper_predictions = 0
    for index, row in enumerate(dataset):
        sample_id = str(first(row, ("id", "sample_id", "qid"), f"test_{index}"))
        if sample_id in ids:
            raise RuntimeError(f"duplicate sample id: {sample_id}")
        ids.add(sample_id)
        question = str(first(row, ("question", "query"), "")).strip()
        gold = first(row, ("golden_answers", "answers", "answer"), [])
        if isinstance(gold, str):
            gold = [gold]
        if not question or not gold:
            raise RuntimeError(f"{sample_id}: missing question or gold answer")

        saved = paper_by_id.get(sample_id, {})
        saved_output = saved.get("output") if isinstance(saved.get("output"), dict) else {}
        prediction = str(first(saved_output, ("pred", "prediction", "answer"), "")).strip()
        if prediction:
            paper_predictions += 1
        output.append(
            {
                "id": sample_id,
                "question": question,
                "golden_answers": [str(item) for item in gold],
                "metadata": row.get("metadata", {}),
                "output": {
                    "pred": prediction,
                    "metric_score": saved_output.get("metric_score", {}),
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    manifest = {
        "dataset_jsonl": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "output": str(output_path),
        "samples": len(output),
        "paper_intermediate": str(paper_path) if paper_path else None,
        "paper_predictions_available": paper_predictions,
        "paper_comparison_available": paper_predictions == len(output),
    }
    manifest_path = output_path.with_name("input_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["paper_comparison_available"]:
        print(
            "WARNING: the original per-item SmartSearch saved predictions were not recovered. "
            "The Qwen3 Blank/SRC/Neutral experiment remains valid, but the paired paper-vs-Blank "
            "comparison must be treated as unavailable.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

