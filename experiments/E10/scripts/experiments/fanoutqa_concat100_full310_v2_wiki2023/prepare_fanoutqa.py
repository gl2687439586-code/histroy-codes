#!/usr/bin/env python3
"""Prepare corrected FanOutQA dev data with generation labels sealed offline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from hp_common import file_sha256, read_json, utc_stamp, write_json, write_jsonl


def walk_evidence(nodes: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(nodes, list):
        for node in nodes:
            yield from walk_evidence(node)
    elif isinstance(nodes, dict):
        evidence = nodes.get("evidence")
        if isinstance(evidence, dict):
            yield evidence
        for key in ("decomposition", "subquestions", "questions"):
            if key in nodes:
                yield from walk_evidence(nodes[key])


def normalize_evidence(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        title = str(item.get("title") or "").strip()
        pageid = item.get("pageid")
        revid = item.get("revid")
        key = (str(pageid or ""), str(revid or ""), title.casefold())
        if not title or key in seen:
            continue
        seen.add(key)
        result.append({
            "title": title,
            "pageid": int(pageid) if str(pageid or "").isdigit() else pageid,
            "revid": int(revid) if str(revid or "").isdigit() else revid,
            "url": item.get("url"),
        })
    return result


def leaf_count(nodes: Any) -> int:
    if isinstance(nodes, list):
        return sum(leaf_count(node) for node in nodes)
    if not isinstance(nodes, dict):
        return 0
    children = None
    for key in ("decomposition", "subquestions", "questions"):
        if isinstance(nodes.get(key), list):
            children = nodes[key]
            break
    return sum(leaf_count(node) for node in children) if children else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_count", type=int, default=310)
    args = parser.parse_args()

    source = Path(args.input)
    rows = read_json(source)
    if not isinstance(rows, list):
        raise ValueError("FanOutQA dev must be a top-level JSON list")
    if args.expected_count and len(rows) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} FanOutQA dev questions; got {len(rows)}"
        )

    questions = []
    labels = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not sample_id or not question or "answer" not in row:
            raise ValueError(f"Invalid FanOutQA row at index {index}")
        evidence = normalize_evidence(
            row.get("necessary_evidence") or list(walk_evidence(row.get("decomposition")))
        )
        if not evidence:
            raise ValueError(f"No necessary_evidence for {sample_id}")
        categories = [str(value) for value in (row.get("categories") or [])]
        questions.append({
            "sample_index": index,
            "sample_id": sample_id,
            "question": question,
            "dataset_name": "fanoutqa",
            "split": "dev",
        })
        labels.append({
            "sample_index": index,
            "sample_id": sample_id,
            "answer": row["answer"],
            "categories": categories,
            "necessary_evidence": evidence,
            "necessary_titles": [item["title"] for item in evidence],
            "necessary_pageids": [item["pageid"] for item in evidence],
            "necessary_evidence_count": len(evidence),
            "decomposition_leaf_count": leaf_count(row.get("decomposition")),
            "dataset_name": "fanoutqa",
            "split": "dev",
        })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    q_path = output / "questions.no_labels.jsonl"
    l_path = output / "gold_labels.sealed.jsonl"
    write_jsonl(q_path, questions)
    write_jsonl(l_path, labels)
    write_json(output / "prepare_manifest.json", {
        "created_at": utc_stamp(),
        "dataset": "FanOutQA",
        "split": "dev",
        "required_release": ">=1.3.0 corrected data",
        "source": str(source),
        "source_sha256": file_sha256(source),
        "records": len(rows),
        "questions_file": str(q_path),
        "labels_file": str(l_path),
        "label_leakage_policy": (
            "Generation reads only questions.no_labels.jsonl. Answers, decomposition "
            "and necessary_evidence remain sealed until trajectories are frozen."
        ),
    })
    print(f"Prepared FanOutQA dev: {len(rows)} questions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
