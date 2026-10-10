#!/usr/bin/env python3
"""Split HotpotQA dev-distractor into label-free questions and sealed gold labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hp_common import canonical_hash, file_sha256, utc_stamp, write_json, write_jsonl


def contexts(row):
    value = row.get("context") or []
    if isinstance(value, dict):
        return list(zip(value.get("title") or [], value.get("sentences") or []))
    return [(item[0], item[1]) for item in value]


def supporting_facts(row):
    value = row.get("supporting_facts") or []
    if isinstance(value, dict):
        return list(zip(value.get("title") or [], value.get("sent_id") or []))
    return [(item[0], item[1]) for item in value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_rows", type=int, default=7405)
    args = parser.parse_args()

    source = Path(args.input)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Official HotpotQA JSON must contain a top-level list")
    if args.expected_rows and len(data) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows; got {len(data)}")

    questions = []
    labels = []
    missing_sentences = []
    seen = set()
    for index, row in enumerate(data):
        sample_id = str(row.get("_id") or row.get("id") or "")
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate id at row {index}: {sample_id}")
        seen.add(sample_id)
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            raise ValueError(f"Missing question/answer for {sample_id}")
        context_map = {str(title): list(sentences) for title, sentences in contexts(row)}
        facts = []
        for title, sentence_id in supporting_facts(row):
            title = str(title)
            sentence_id = int(sentence_id)
            sentences = context_map.get(title) or []
            sentence = str(sentences[sentence_id]) if 0 <= sentence_id < len(sentences) else ""
            if not sentence:
                missing_sentences.append([sample_id, title, sentence_id])
            facts.append({
                "title": title, "sent_id": sentence_id, "sentence": sentence,
            })
        questions.append({
            "sample_index": index,
            "sample_id": sample_id,
            "question": question,
            "type": str(row.get("type") or "unknown"),
            "level": str(row.get("level") or "unknown"),
        })
        labels.append({
            "sample_index": index,
            "sample_id": sample_id,
            "answer": answer,
            "gold_answers": [answer],
            "type": str(row.get("type") or "unknown"),
            "level": str(row.get("level") or "unknown"),
            "supporting_facts": facts,
            "supporting_titles": sorted({fact["title"] for fact in facts}),
        })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "questions.no_labels.jsonl", questions)
    write_jsonl(output / "gold_labels.sealed.jsonl", labels)
    manifest = {
        "created_at": utc_stamp(),
        "benchmark": "hotpotqa",
        "source_split": "official dev distractor labels; open-domain retrieval",
        "source_path": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "rows": len(data),
        "unique_ids": len(seen),
        "missing_supporting_sentence_text": len(missing_sentences),
        "missing_examples": missing_sentences[:20],
        "generation_file": "questions.no_labels.jsonl",
        "offline_only_label_file": "gold_labels.sealed.jsonl",
        "questions_content_hash": canonical_hash(questions),
        "labels_content_hash": canonical_hash(labels),
        "label_leakage_policy": (
            "Generation scripts read questions.no_labels.jsonl only. Gold answers, "
            "supporting titles and sentences are joined only by offline analysis."
        ),
    }
    write_json(output / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

