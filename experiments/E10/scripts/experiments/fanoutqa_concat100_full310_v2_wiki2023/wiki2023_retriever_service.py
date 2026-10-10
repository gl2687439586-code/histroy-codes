#!/usr/bin/env python3
"""FlashRAG-compatible Top-k passage service over FanOutQA's 2023-09 Kiwix mirror.

Kiwix retrieves Wikipedia *pages*.  The previous SmartSearch service retrieves
roughly 100-word passages.  To keep the observation granularity comparable, this
adapter deterministically selects one query-relevant <=100-word passage from each
of the top-ranked Kiwix pages and exposes the existing ``/health`` and ``/search``
HTTP contract.  It never reads FanOutQA gold evidence labels.
"""

import argparse
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from xml.etree import ElementTree

BACKEND_ID = "fanoutqa_wikipedia_en_all_nopic_2023-09_kiwix_passage100_upto5"
WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
SPACE_RE = re.compile(r"[ \t]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with",
}


def normalize_markdown(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    result: List[str] = []
    previous_blank = True
    for line in lines:
        if line:
            result.append(line)
            previous_blank = False
        elif not previous_blank:
            result.append("")
            previous_blank = True
    return "\n".join(result).strip()


def query_terms(query: str) -> List[str]:
    terms = [match.group(0).casefold() for match in WORD_RE.finditer(query)]
    useful = [term for term in terms if term not in STOPWORDS and len(term) > 1]
    return useful or terms


def word_windows(text: str, size: int = 100, stride: int = 50) -> Iterable[tuple[int, str]]:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return
    if len(matches) <= size:
        yield 0, text.strip()
        return
    starts = list(range(0, len(matches), stride))
    final_start = max(0, len(matches) - size)
    if final_start not in starts:
        starts.append(final_start)
    for start in sorted(set(starts)):
        end = min(len(matches), start + size)
        char_start = matches[start].start()
        char_end = matches[end - 1].end()
        yield start, text[char_start:char_end].strip()
        if end == len(matches):
            break


def best_passage(title: str, markdown: str, query: str) -> str:
    """Choose a stable, query-relevant passage without consulting gold labels."""
    text = normalize_markdown(markdown)
    if not text:
        return "This page does not contain extractable text."
    terms = query_terms(query)
    title_folded = title.casefold()
    best: tuple[tuple[int, int, int, int], str] | None = None
    for start, candidate in word_windows(text):
        folded = candidate.casefold()
        present = sum(1 for term in set(terms) if term in folded)
        occurrences = sum(folded.count(term) for term in terms)
        title_overlap = sum(1 for term in set(terms) if term in title_folded)
        # More unique query terms is primary; earlier text breaks exact ties.
        score = (present, occurrences, title_overlap, -start)
        if best is None or score > best[0]:
            best = (score, candidate)
    assert best is not None
    return best[1]


def parse_kiwix_search_xml(raw: bytes, evidence_type: Any) -> List[Any]:
    """Parse Kiwix RSS 2.0 and Atom search result documents."""

    root = ElementTree.fromstring(raw)
    pages: List[Any] = []
    for candidate in root.iter():
        container = candidate.tag.rsplit("}", 1)[-1]
        if container not in {"entry", "item"}:
            continue
        title = ""
        url = ""
        for child in candidate:
            field = child.tag.rsplit("}", 1)[-1]
            if field == "title" and not title:
                title = str(child.text or "").strip()
            elif field == "link" and not url:
                # Atom stores the URL in href; RSS 2.0 uses element text.
                url = str(child.get("href") or child.text or "").strip()
        if title and url:
            pages.append(evidence_type(
                title=title, pageid=0, revid=0, url=url
            ))
    return pages


class KiwixPassageRetriever:
    def __init__(self, cache_path: Path):
        if os.getenv("FANOUTQA_WIKIPEDIA_TYPE") != "kiwix":
            raise RuntimeError("FANOUTQA_WIKIPEDIA_TYPE must be kiwix")
        self.kiwix_base = os.environ["FANOUTQA_KIWIX_BASE"].rstrip("/")
        self.zim_name = os.environ["FANOUTQA_KIWIX_ZIMNAME"]
        from fanoutqa import wiki_content
        from fanoutqa.models import Evidence
        from fanoutqa.wiki import KIWIX_CACHE_DIR

        self.wiki_content = wiki_content
        self.evidence_type = Evidence
        # fanoutqa.wiki defines KIWIX_CACHE_DIR but upstream currently does not
        # create it before Path.write_text().  Initialise it explicitly so the
        # first page fetch works on a clean server.
        self.upstream_page_cache = Path(KIWIX_CACHE_DIR).resolve()
        self.upstream_page_cache.mkdir(parents=True, exist_ok=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path), timeout=120, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS results ("
            "cache_key TEXT PRIMARY KEY, query TEXT NOT NULL, top_n INTEGER NOT NULL, "
            "payload TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.db.commit()
        self.lock = threading.RLock()
        self.page_locks: Dict[str, threading.Lock] = {}

    def wiki_search_xml(self, query: str, results: int) -> List[Any]:
        """Search Kiwix 3.8+ using its explicit machine-readable XML mode.

        Kiwix defaults ``/search`` to HTML.  Current Kiwix 3.8 returns RSS 2.0
        for ``format=xml`` while FanOutQA's helper assumes Atom.  This adapter
        accepts both without changing the ranking order.
        """
        import requests

        response = requests.get(
            self.kiwix_base + "/search",
            params={
                "pattern": query,
                "start": 0,
                "pageLength": results,
                "books.name": self.zim_name,
                "format": "xml",
            },
            timeout=120,
        )
        response.raise_for_status()
        try:
            pages = parse_kiwix_search_xml(response.content, self.evidence_type)
        except ElementTree.ParseError as exc:
            preview = response.text[:200].replace("\n", " ")
            raise RuntimeError(
                "Kiwix did not return parseable XML despite format=xml; "
                f"content_type={response.headers.get('content-type')!r}, "
                f"preview={preview!r}"
            ) from exc

        return pages

    def page_content(self, page: Any, identity: str) -> str:
        # fanoutqa's disk page cache writes directly to the target file.  A
        # per-page lock prevents concurrent workers from observing a partial
        # cache write while still allowing different pages to load in parallel.
        with self.lock:
            page_lock = self.page_locks.setdefault(identity, threading.Lock())
        with page_lock:
            return self.wiki_content(page)

    def health(self) -> Dict[str, Any]:
        import requests

        response = requests.get(self.kiwix_base + "/", timeout=10)
        response.raise_for_status()
        return {
            "status": "healthy",
            "backend": BACKEND_ID,
            "snapshot": "2023-09",
            "zim_name": self.zim_name,
            "kiwix_base": self.kiwix_base,
            "upstream_page_cache": str(self.upstream_page_cache),
            "passage_policy": "best_query_lexical_window_max_100_words_per_page",
            "result_cardinality_policy": (
                "natural_exact_query_zero_to_requested_max_no_backfill"
            ),
            "gold_labels_loaded": False,
        }

    def search(self, query: str, top_n: int) -> List[Dict[str, Any]]:
        query = " ".join(str(query or "").split())
        if not query:
            raise ValueError("Empty query")
        if not 1 <= top_n <= 50:
            raise ValueError("top_n must be in [1, 50]")
        key = json.dumps([BACKEND_ID, query, top_n], ensure_ascii=False)
        with self.lock:
            found = self.db.execute(
                "SELECT payload FROM results WHERE cache_key=?", (key,)
            ).fetchone()
        if found:
            return json.loads(found[0])

        # Kiwix is sparse rather than dense: Top-k is a maximum, not a promise.
        # Run exactly the model's natural query and return 0..top_n usable pages
        # in Kiwix order.  Never broaden, rewrite, duplicate, or backfill it.
        documents: List[Dict[str, Any]] = []
        seen = set()
        page_errors: List[str] = []
        pages = self.wiki_search_xml(query, results=max(20, top_n * 4))
        for source_rank, page in enumerate(pages, 1):
            title = str(getattr(page, "title", "") or "").strip()
            url = str(getattr(page, "url", "") or "").strip()
            identity = (title.casefold(), url)
            if not title or identity in seen:
                continue
            seen.add(identity)
            try:
                markdown = self.page_content(page, url or title)
            except Exception as exc:
                page_errors.append(f"{title}:{type(exc).__name__}:{exc}")
                continue
            passage = best_passage(title, markdown, query)
            contents = f'"{title}"\n{passage}'
            documents.append({
                "id": url or title,
                "contents": contents,
                "title": title,
                "url": url,
                "pageid": int(getattr(page, "pageid", 0) or 0),
                "revid": int(getattr(page, "revid", 0) or 0),
                "corpus_snapshot": "2023-09",
                "retriever_backend": BACKEND_ID,
                "requested_query": query,
                "retrieval_query": query,
                "retrieval_query_stage": "exact_natural_query",
                "retrieval_query_is_fallback": False,
                "retrieval_rank_within_query": source_rank,
                "retrieval_rank_global": len(documents) + 1,
            })
            if len(documents) == top_n:
                break
        for document in documents:
            document["requested_max_k"] = top_n
            document["actual_retrieved_k"] = len(documents)
            document["skipped_page_error_count"] = len(page_errors)
        encoded = json.dumps(documents, ensure_ascii=False, separators=(",", ":"))
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO results(cache_key,query,top_n,payload,created_at) "
                "VALUES (?,?,?,?,?)", (key, query, top_n, encoded, time.time())
            )
        return documents


def create_app(retriever: KiwixPassageRetriever):
    """Create the HTTP layer lazily so offline tests need no FastAPI install."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class QueryRequest(BaseModel):
        query: str
        top_n: int = 5
        return_score: bool = False

    app = FastAPI()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        try:
            return retriever.health()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/search")
    def search(request: QueryRequest):
        try:
            documents = retriever.search(request.query, request.top_n)
            if request.return_score:
                return documents, [0.0] * len(documents)
            return documents
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument(
        "--cache",
        default="/home/researcher/smartsearch_rebuild/retrieval/"
                "fanoutqa_wiki2023/retriever_cache.sqlite",
    )
    args = parser.parse_args()
    retriever = KiwixPassageRetriever(Path(args.cache))
    import uvicorn

    uvicorn.run(create_app(retriever), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
