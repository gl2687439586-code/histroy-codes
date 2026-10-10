#!/usr/bin/env python3
"""Write the existing E5/wiki18 FlashRAG Top-5 configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    values = {
        name: Path(value).resolve()
        for name, value in {
            "model": args.model,
            "index": args.index,
            "corpus": args.corpus,
        }.items()
    }
    for name, path in values.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    config = {
        "gpu_id": None,
        "retrieval_method": str(values["model"]),
        "retrieval_model_path": str(values["model"]),
        "index_path": str(values["index"]),
        "faiss_gpu": False,
        "corpus_path": str(values["corpus"]),
        "retrieval_topk": 5,
        "retrieval_batch_size": 8,
        "retrieval_query_max_length": 512,
        "retrieval_use_fp16": True,
        "retrieval_pooling_method": "mean",
        "use_sentence_transformer": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

