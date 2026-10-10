#!/usr/bin/env python3
"""Shared retriever client, query cache, and configurable Top-k renderer."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hp_common import (
    EXPOSURE_STEP_TOKENS, MAX_BOUNDARY_BACKOFF_TOKENS, TOP_K,
    ExactTokenizer, canonical_hash,
    extract_title, sha256_text,
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


def _token_span_for_chars(offsets: List[Tuple[int, int]],
                          start: int, end: int) -> Tuple[int, int]:
    """Map a character interval to a 0-based half-open token interval."""
    touched = [index for index, (left, right) in enumerate(offsets)
               if right > start and left < end]
    if not touched:
        insertion = sum(1 for _, right in offsets if right <= start)
        return insertion, insertion
    return touched[0], touched[-1] + 1


def concatenate_documents(tokenizer: ExactTokenizer,
                          documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate complete retrieved passages before applying any budget.

    The only between-document material is two newlines.  It is part of the
    evidence context and therefore part of the exact Qwen token accounting.
    """
    documents = enrich_documents(tokenizer, documents)
    chunks: List[str] = []
    char_spans: List[Tuple[int, int]] = []
    cursor = 0
    for index, document in enumerate(documents):
        if index:
            chunks.append("\n\n")
            cursor += 2
        start = cursor
        contents = str(document["contents"])
        chunks.append(contents)
        cursor += len(contents)
        char_spans.append((start, cursor))
    full_text = "".join(chunks)
    token_ids = tokenizer.encode(full_text)
    offsets = tokenizer.offsets(full_text)
    if len(offsets) != len(token_ids):
        raise RuntimeError(
            f"Tokenizer offset mismatch: ids={len(token_ids)} offsets={len(offsets)}"
        )
    doc_rows = []
    for rank, (document, (char_start, char_end)) in enumerate(
        zip(documents, char_spans), 1
    ):
        token_start, token_end = _token_span_for_chars(
            offsets, char_start, char_end
        )
        doc_rows.append({
            "rank": rank,
            "id": document["id"],
            "title": document["title"],
            "contents": document["contents"],
            "contents_sha256": document["contents_sha256"],
            "raw_token_count": document["raw_token_count"],
            "context_char_start_0based": char_start,
            "context_char_end_exclusive_0based": char_end,
            "context_token_start_0based": token_start,
            "context_token_end_exclusive_0based": token_end,
        })
    return {
        "full_context_text": full_text,
        "full_context_sha256": sha256_text(full_text),
        "full_context_token_ids": token_ids,
        "full_context_offsets": offsets,
        "full_context_token_count": len(token_ids),
        "documents": doc_rows,
        "separator": "\n\n",
    }


def exposure_conditions(tokenizer: ExactTokenizer,
                        documents: List[Dict[str, Any]],
                        step: int = EXPOSURE_STEP_TOKENS) -> List[Dict[str, Any]]:
    """Return 100-token caps plus one exact, per-search FULL endpoint."""
    if step <= 0:
        raise ValueError("Exposure step must be positive")
    total = concatenate_documents(tokenizer, documents)["full_context_token_count"]
    if total <= 0:
        raise ValueError("Concatenated evidence context is empty")
    result = [{
        "condition_id": f"c{cap:05d}",
        "condition_label": str(cap),
        "exposure_mode": "token_cap",
        "token_cap_per_search": cap,
        "root_exposure_tokens": cap,
        "root_full_context_tokens": total,
        "step_tokens": step,
    } for cap in range(step, total, step)]
    result.append({
        "condition_id": f"full{total:05d}",
        "condition_label": "FULL",
        "exposure_mode": "full_per_search",
        "token_cap_per_search": None,
        "root_exposure_tokens": total,
        "root_full_context_tokens": total,
        "step_tokens": step,
    })
    return result


def largest_roundtrippable_prefix(tokenizer: ExactTokenizer,
                                  token_ids: List[int],
                                  full_text: str,
                                  target_tokens: int,
                                  offsets: List[Tuple[int, int]] | None = None,
                                  ) -> Tuple[str, List[int], int, int]:
    """Return the largest exact token prefix not exceeding ``target_tokens``.

    A byte-level tokenizer prefix can end inside a character representation.
    Decoding arbitrary prefix ids may also normalize text even when cleanup is
    disabled.  Therefore candidates must be sliced from the *original evidence
    text* using the fast tokenizer's character offsets, never reconstructed
    from ``decode(prefix_ids)``.  The candidate is then re-encoded and accepted
    only when it exactly equals the selected id prefix.
    """
    total = len(token_ids)
    target = max(0, min(int(target_tokens), total))
    if target <= 0:
        raise ValueError("Evidence token target must be positive")
    if target == total:
        reencoded = tokenizer.encode(full_text)
        if reencoded != token_ids:
            raise RuntimeError("Full evidence context does not round-trip exactly")
        return full_text, reencoded, total, 0
    token_offsets = offsets if offsets is not None else tokenizer.offsets(full_text)
    if len(token_offsets) != total:
        raise RuntimeError(
            f"Tokenizer offset mismatch: ids={total} offsets={len(token_offsets)}"
        )
    for actual in range(target, 0, -1):
        char_end = int(token_offsets[actual - 1][1])
        if not 0 < char_end <= len(full_text):
            continue
        candidate = full_text[:char_end]
        reencoded = tokenizer.encode(candidate)
        if reencoded == token_ids[:actual]:
            return candidate, reencoded, actual, target - actual
    raise RuntimeError(
        "No non-empty round-trippable evidence prefix exists within the token cap"
    )


def render_evidence(tokenizer: ExactTokenizer, documents: List[Dict[str, Any]],
                    condition: Dict[str, Any], search_round: int) -> Dict[str, Any]:
    """Expose a global prefix of the complete concatenated evidence context."""
    context = concatenate_documents(tokenizer, documents)
    total = int(context["full_context_token_count"])
    mode = str(condition["exposure_mode"])
    if mode == "full_per_search":
        requested_cap = total
        target_cap = total
    elif mode == "token_cap":
        requested_cap = int(condition["token_cap_per_search"])
        target_cap = min(requested_cap, total)
    else:
        raise ValueError(f"Unknown exposure mode: {mode}")
    token_ids = context["full_context_token_ids"]
    visible_context, roundtrip_ids, exposed, boundary_backoff = (
        largest_roundtrippable_prefix(
            tokenizer, token_ids, context["full_context_text"], target_cap,
            context["full_context_offsets"],
        )
    )
    if boundary_backoff > MAX_BOUNDARY_BACKOFF_TOKENS:
        raise RuntimeError(
            "Evidence boundary backoff exceeds safety limit: "
            f"target={target_cap}, actual={exposed}, "
            f"backoff={boundary_backoff}, "
            f"limit={MAX_BOUNDARY_BACKOFF_TOKENS}"
        )
    offsets = context["full_context_offsets"]
    visible_char_end = len(visible_context)
    visible_passages = []
    actual_by_rank = [0] * TOP_K
    for document in context["documents"]:
        token_start = int(document["context_token_start_0based"])
        token_end = int(document["context_token_end_exclusive_0based"])
        overlap_end = min(token_end, exposed)
        actual = max(0, overlap_end - token_start)
        actual_by_rank[int(document["rank"]) - 1] = actual
        if actual <= 0:
            continue
        char_start = int(document["context_char_start_0based"])
        char_end = min(int(document["context_char_end_exclusive_0based"]), visible_char_end)
        visible_text = context["full_context_text"][char_start:char_end]
        visible_passages.append({
            "rank": document["rank"],
            "id": document["id"],
            "title": document["title"],
            "contents_sha256": document["contents_sha256"],
            "actual_visible_tokens": actual,
            "raw_token_count": document["raw_token_count"],
            "visible_text": visible_text,
            "visible_text_sha256": sha256_text(visible_text),
            "context_token_start_0based": token_start,
            "context_token_end_exclusive_0based": overlap_end,
        })
    header = (
        f"[Search round={search_round}; concatenated Top-{TOP_K} evidence; "
        f"visible_tokens={exposed}/{total}; mode={mode}]\n"
    )
    evidence_text = header + visible_context
    return {
        "condition_id": condition["condition_id"],
        "condition_label": condition["condition_label"],
        "exposure_mode": mode,
        "requested_token_cap": requested_cap,
        "target_token_cap_before_roundtrip": target_cap,
        "full_context_token_count": total,
        "actual_exposed_token_count": exposed,
        "visible_context_reencoded_token_count": len(roundtrip_ids),
        "exact_token_roundtrip": True,
        "prefix_boundary_policy": (
            "largest_exact_original_text_offset_prefix_not_exceeding_cap"
        ),
        "roundtrip_boundary_backoff_tokens": boundary_backoff,
        "actual_tokens_by_rank": actual_by_rank,
        "visible_evidence_tokens": exposed,
        "underfilled_tokens": max(0, requested_cap - exposed),
        "visible_passages": visible_passages,
        "visible_context_text": visible_context,
        "visible_context_sha256": sha256_text(visible_context),
        "evidence_text": evidence_text,
        "evidence_text_sha256": sha256_text(evidence_text),
        "full_context_sha256": context["full_context_sha256"],
        "context_separator": context["separator"],
        "retrieved_docs": context["documents"],
        "retrieval_hash": canonical_hash([
            [doc["id"], doc["contents_sha256"]] for doc in context["documents"]
        ]),
    }
