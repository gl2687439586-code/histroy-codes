#!/usr/bin/env python3
"""Create a deterministic random FanOutQA dev subset without label leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hp_common import (
    canonical_hash, file_sha256, read_jsonl, utc_stamp, write_json, write_jsonl,
)


def sample_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    labels = {str(row["sample_id"]): row for row in read_jsonl(args.labels)}
    if not 0 < args.sample_size <= len(questions):
        raise ValueError(
            f"sample_size must be in [1,{len(questions)}], got {args.sample_size}"
        )
    ranked = sorted(
        questions,
        key=lambda row: (sample_rank(args.seed, str(row["sample_id"])),
                         str(row["sample_id"])),
    )
    sampled = ranked[:args.sample_size]
    sampled_ids = [str(row["sample_id"]) for row in sampled]
    missing = [sample_id for sample_id in sampled_ids if sample_id not in labels]
    if missing:
        raise ValueError(f"Selected questions missing sealed labels: {missing[:10]}")

    # Restore source order after random selection.  Sharding and output order
    # are then stable and easy to compare with the official split.
    sampled.sort(key=lambda row: int(row.get("sample_index", 0)))
    sampled_labels = [labels[str(row["sample_id"])] for row in sampled]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    question_out = output / "questions.no_labels.jsonl"
    label_out = output / "gold_labels.sealed.jsonl"
    write_jsonl(question_out, sampled)
    write_jsonl(label_out, sampled_labels)
    manifest = {
        "created_at": utc_stamp(),
        "method": "lowest SHA256(seed\\0sample_id), then restore official source order",
        "seed": args.seed,
        "sample_size": args.sample_size,
        "source_question_count": len(questions),
        "questions_source_sha256": file_sha256(args.questions),
        "labels_source_sha256": file_sha256(args.labels),
        "selected_sample_ids": [str(row["sample_id"]) for row in sampled],
        "selected_ids_sha256": canonical_hash(
            [str(row["sample_id"]) for row in sampled]
        ),
        "generation_file": str(question_out),
        "offline_only_label_file": str(label_out),
        "label_leakage_policy": (
            "Generation reads only questions.no_labels.jsonl. The paired sealed "
            "labels are consumed only after trajectories are frozen."
        ),
    }
    write_json(output / "sample_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
