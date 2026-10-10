#!/usr/bin/env python3
"""Write a CPU-FAISS retriever config without shell-quoting path hazards."""

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

    model = Path(args.model).resolve()
    index = Path(args.index).resolve()
    corpus = Path(args.corpus).resolve()
    for path in (model, index, corpus):
        if not path.exists():
            raise FileNotFoundError(path)
    config = {
        # CUDA visibility is controlled by the tmux launcher. Keeping this null
        # prevents FlashRAG Config from remapping the process back to GPU 0.
        "gpu_id": None,
        "retrieval_method": str(model),
        "retrieval_model_path": str(model),
        "index_path": str(index),
        "faiss_gpu": False,
        "corpus_path": str(corpus),
        "retrieval_topk": 5,
        "retrieval_batch_size": 8,
        "retrieval_query_max_length": 512,
        "retrieval_use_fp16": True,
        "retrieval_pooling_method": "mean",
        "use_sentence_transformer": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

