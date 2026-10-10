#!/usr/bin/env python3
"""Shared retriever client, query cache, and Top-10 evidence renderer."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hp_common import (
    TOP_K, ExactTokenizer, canonical_hash, extract_title,
    requested_tokens_by_rank, sha256_text,
)


class CachedRetriever:
    def __init__(self, base_url: str, cache_path: str | Path,
                 timeout: float = 300.0, retries: int = 3):
        self.url = base_url.rstrip("/") + "/search"
        self.timeout = timeout
        self.retries = retries
        self.db = sqlite3.connect(str(cache_path), timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS retrieval_cache "
            "(cache_key TEXT PRIMARY KEY, query TEXT NOT NULL, top_k INTEGER NOT NULL, "
            "payload TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.db.commit()

    def search(self, query: str, top_k: int = TOP_K) -> Tuple[List[Dict[str, Any]], bool, float]:
        import requests

        query = " ".join(str(query or "").split())
        if not query:
            raise ValueError("Empty search query")
        cache_key = canonical_hash({"query": query, "top_k": top_k})
        found = self.db.execute(
            "SELECT payload FROM retrieval_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
        if found:
            return json.loads(found[0]), True, 0.0
        last_error: Exception | None = None
        for attempt in range(self.retries):
            started = time.time()
            try:
                response = requests.post(
                    self.url,
                    json={"query": query, "top_n": top_k, "return_score": False},
                    timeout=self.timeout,
                )
                if not response.ok:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:1000]}"
                    )
                payload = response.json()
                if isinstance(payload, dict):
                    payload = payload.get("documents") or payload.get("result") or payload.get("data")
                if not isinstance(payload, list) or len(payload) < top_k:
                    raise RuntimeError(
                        f"Retriever returned {len(payload) if isinstance(payload, list) else type(payload)}; "
                        f"Top-{top_k} required"
                    )
                docs = []
                for rank, item in enumerate(payload[:top_k], 1):
                    contents = str(item.get("contents") or item.get("content") or item.get("text") or "")
                    if not contents:
                        raise RuntimeError(f"Empty contents at rank {rank}")
                    docs.append({
                        "rank": rank,
                        "id": str(item.get("id") or item.get("key") or ""),
                        "contents": contents,
                        "title": extract_title(contents),
                        "contents_sha256": sha256_text(contents),
                    })
                encoded = json.dumps(docs, ensure_ascii=False, separators=(",", ":"))
                with self.db:
                    self.db.execute(
                        "INSERT OR IGNORE INTO retrieval_cache"
                        "(cache_key,query,top_k,payload,created_at) VALUES (?,?,?,?,?)",
                        (cache_key, query, top_k, encoded, time.time()),
                    )
                return docs, False, time.time() - started
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Retriever failed after {self.retries} tries: {last_error}")

    def close(self) -> None:
        self.db.close()


def enrich_documents(tokenizer: ExactTokenizer,
                     documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for document in documents[:TOP_K]:
        item = dict(document)
        item["raw_token_count"] = tokenizer.count(item["contents"])
        item["title"] = item.get("title") or extract_title(item["contents"])
        item["contents_sha256"] = item.get("contents_sha256") or sha256_text(item["contents"])
        enriched.append(item)
    if len(enriched) != TOP_K:
        raise ValueError(f"Expected exactly Top-{TOP_K}; got {len(enriched)}")
    return enriched


def render_evidence(tokenizer: ExactTokenizer, documents: List[Dict[str, Any]],
                    budget: int, search_round: int) -> Dict[str, Any]:
    documents = enrich_documents(tokenizer, documents)
    allocation = requested_tokens_by_rank(budget)
    parts = []
    visible = []
    actual_by_rank = [0] * TOP_K
    for rank, (document, limit) in enumerate(zip(documents, allocation), 1):
        if not limit:
            continue
        token_ids = tokenizer.encode(document["contents"])
        visible_text = tokenizer.decode(token_ids[:limit])
        actual = min(limit, len(token_ids))
        separator = "\n\n" if parts else ""
        wrapper = (
            f"[Search round={search_round} rank={rank} id={document['id']} "
            f"title={json.dumps(document['title'], ensure_ascii=False)} "
            f"visible_prefix={limit}]\n"
        )
        parts.append(separator + wrapper + visible_text)
        actual_by_rank[rank - 1] = actual
        visible.append({
            "rank": rank,
            "id": document["id"],
            "title": document["title"],
            "contents_sha256": document["contents_sha256"],
            "requested_visible_tokens": limit,
            "actual_visible_tokens": actual,
            "raw_token_count": len(token_ids),
            "visible_text": visible_text,
            "visible_text_sha256": sha256_text(visible_text),
        })
    evidence_text = "".join(parts)
    return {
        "budget": budget,
        "allocation": allocation,
        "actual_tokens_by_rank": actual_by_rank,
        "visible_evidence_tokens": sum(actual_by_rank),
        "underfilled_tokens": budget - sum(actual_by_rank),
        "visible_passages": visible,
        "evidence_text": evidence_text,
        "evidence_text_sha256": sha256_text(evidence_text),
        "retrieved_docs": [
            {
                "rank": rank,
                "id": doc["id"],
                "title": doc["title"],
                "contents_sha256": doc["contents_sha256"],
                "raw_token_count": doc["raw_token_count"],
            }
            for rank, doc in enumerate(documents, 1)
        ],
        "retrieval_hash": canonical_hash([
            [doc["id"], doc["contents_sha256"]] for doc in documents
        ]),
    }
