"""Join the uploaded 500 IDs to official HotpotQA, without changing membership."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_hotpotqa import contexts, supporting_facts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_queries(path, expected=500):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [str(row.get("query_id", "")) for row in rows]
    if len(rows) != expected or len(set(ids)) != expected or any(not sid for sid in ids):
        raise RuntimeError(f"Expected {expected} unique query IDs; records={len(rows)}, unique={len(set(ids))}")
    if any(set(row) != {"query_id", "query"} or not isinstance(row["query"], str) or not row["query"].strip() for row in rows):
        raise RuntimeError("Queries must have exactly query_id and a nonempty query")
    return rows


def join_official(queries, official):
    by_id = {str(row["_id"]): row for row in official}
    if len(by_id) != len(official):
        raise RuntimeError("Official dataset has duplicate IDs")
    questions, labels = [], []
    for index, query in enumerate(queries):
        sid = str(query["query_id"])
        row = by_id.get(sid)
        # Original prepare_hotpotqa.py strips outer whitespace in official text.
        # Keep the uploaded text intact; only compare against that same rule.
        if row is None or query["query"] != row["question"].strip():
            raise RuntimeError(f"Missing ID or nonidentical official question: {sid}")
        common = {"sample_index": index, "sample_id": sid,
                  "type": row.get("type", "unknown"), "level": row.get("level", "unknown")}
        questions.append({**common, "question": query["query"]})
        context_map = {str(title): list(sentences) for title, sentences in contexts(row)}
        facts = []
        for title, sentence_id in supporting_facts(row):
            sentences = context_map.get(str(title), [])
            sentence_id = int(sentence_id)
            if not 0 <= sentence_id < len(sentences):
                raise RuntimeError(f"Missing gold supporting sentence for {sid}: {title}/{sentence_id}")
            facts.append({"title": str(title), "sent_id": sentence_id, "sentence": str(sentences[sentence_id])})
        labels.append({**common, "answer": row["answer"], "gold_answers": [row["answer"]],
                       "supporting_facts": facts,
                       "supporting_titles": sorted({fact["title"] for fact in facts})})
    return questions, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--official-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    queries = load_queries(args.query_file)
    official = json.loads(args.official_json.read_text(encoding="utf-8"))
    questions, labels = join_official(queries, official)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("questions.no_labels.jsonl", questions), ("gold_labels.sealed.jsonl", labels)):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"records": len(questions), "query_file_sha256": sha256(args.query_file),
                "official_json_sha256": sha256(args.official_json),
                "question_text_matches_original_prepare_strip_rule": True,
                "uploaded_question_text_preserved": True, "ordering": "uploaded query-file order",
                "no_generation_gold_fields": True,
                "files": {name: sha256(args.output_dir / name) for name in
                          ("questions.no_labels.jsonl", "gold_labels.sealed.jsonl")}}
    (args.output_dir / "INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
