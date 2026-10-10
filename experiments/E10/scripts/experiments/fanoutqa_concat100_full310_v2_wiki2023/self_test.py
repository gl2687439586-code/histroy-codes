#!/usr/bin/env python3
"""Offline protocol tests; no model, retriever, GPU, or labels are queried."""

from __future__ import annotations

import tempfile
from pathlib import Path

from actions import parse_action
from compare_blank_budget import paired_delta
from hp_common import ExactTokenizer
from prompts import decision_prompt, forced_answer_prompt, initial_search_prompt
from prepare_fanoutqa import leaf_count, normalize_evidence, walk_evidence
from retrieval import (
    blank_condition, concatenate_documents, exposure_conditions, render_evidence,
)
from run_trajectories import conditions_for_root
from wiki2023_retriever_service import (
    BACKEND_ID, best_passage, create_app, parse_kiwix_search_xml,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in str(text)]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(value) for value in ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def docs():
    return [
        {
            "rank": rank,
            "id": str(rank),
            "title": f"Page {rank}",
            "contents": f'"Page {rank}"\n' + chr(64 + rank) * 52,
            "contents_sha256": "",
        }
        for rank in range(1, 6)
    ]


def main() -> int:
    tokenizer = ExactTokenizer(tokenizer=CharacterTokenizer())
    documents = docs()
    context = concatenate_documents(tokenizer, documents)
    conditions = exposure_conditions(tokenizer, documents)
    assert [row["root_exposure_tokens"] for row in conditions[:-1]] == [
        100, 200, 300
    ]
    assert conditions[-1]["condition_label"] == "FULL"
    assert conditions[-1]["root_exposure_tokens"] == context["full_context_token_count"]
    budget = render_evidence(tokenizer, documents, conditions[1], 1)
    assert budget["actual_exposed_token_count"] == 200
    assert budget["exact_token_roundtrip"] is True
    blank = blank_condition(tokenizer, documents)
    blank_rendered = render_evidence(tokenizer, documents, blank, 1)
    assert blank_rendered["actual_exposed_token_count"] == context["full_context_token_count"]
    assert blank_rendered["evidence_text"].startswith("result:\n")

    two_documents = documents[:2]
    two_context = concatenate_documents(tokenizer, two_documents)
    two_budget = exposure_conditions(tokenizer, two_documents)
    assert 0 < two_context["full_context_token_count"]
    assert two_budget[-1]["root_exposure_tokens"] == two_context[
        "full_context_token_count"
    ]
    two_rendered = render_evidence(tokenizer, two_documents, two_budget[0], 1)
    assert two_rendered["requested_max_k"] == 5
    assert two_rendered["actual_retrieved_k"] == 2
    assert two_rendered["retrieval_shortfall"] is True

    zero_budget = exposure_conditions(tokenizer, [])
    assert zero_budget == [{
        "condition_id": "full00000",
        "condition_label": "FULL",
        "exposure_mode": "full_per_search",
        "token_cap_per_search": None,
        "root_exposure_tokens": 0,
        "root_full_context_tokens": 0,
        "step_tokens": 100,
    }]
    zero_rendered = render_evidence(tokenizer, [], zero_budget[0], 1)
    assert zero_rendered["actual_retrieved_k"] == 0
    assert zero_rendered["actual_exposed_token_count"] == 0
    assert zero_rendered["zero_result"] is True
    assert "No documents retrieved" in zero_rendered["evidence_text"]

    root = {
        "sample_id": "fanout-test",
        "root_full_context_tokens": context["full_context_token_count"],
        "exposure_conditions": conditions,
    }
    assert len(conditions_for_root(root, "", "budget")) == len(conditions)
    only_blank = conditions_for_root(root, "", "blank")
    assert len(only_blank) == 1
    assert only_blank[0]["condition_label"] == "BLANK"

    decomposition = [{
        "id": "a", "question": "leaf", "decomposition": [],
        "evidence": {
            "pageid": 7, "revid": 8, "title": "Evidence Page", "url": "u"
        },
    }]
    extracted = normalize_evidence(walk_evidence(decomposition))
    assert extracted == [{
        "title": "Evidence Page", "pageid": 7, "revid": 8, "url": "u"
    }]
    assert leaf_count(decomposition) == 1

    selected = best_passage(
        "Target Page",
        "irrelevant words " * 120 + "rareterm supporting fact " * 8,
        "rareterm",
    )
    assert "rareterm" in selected
    assert len(selected.split()) <= 100
    assert "2023-09" in BACKEND_ID
    assert "upto5" in BACKEND_ID

    class EvidenceStub:
        def __init__(self, title, pageid, revid, url):
            self.title = title
            self.pageid = pageid
            self.revid = revid
            self.url = url

    rss = b'''<?xml version="1.0"?>
    <rss version="2.0"><channel><item><title>RSS title</title>
    <link>/content/book/A/RSS_title</link></item></channel></rss>'''
    atom = b'''<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom title</title>
    <link href="/content/book/A/Atom_title"/></entry></feed>'''
    rss_pages = parse_kiwix_search_xml(rss, EvidenceStub)
    atom_pages = parse_kiwix_search_xml(atom, EvidenceStub)
    assert [(page.title, page.url) for page in rss_pages] == [
        ("RSS title", "/content/book/A/RSS_title")
    ]
    assert [(page.title, page.url) for page in atom_pages] == [
        ("Atom title", "/content/book/A/Atom_title")
    ]

    class RetrieverStub:
        def health(self):
            return {"status": "healthy"}

        def search(self, query, top_n):
            return [{"id": str(index)} for index in range(top_n)]

    # Regression test for FastAPI/Pydantic: the locally defined QueryRequest
    # must be represented as a JSON request body, not a required query string.
    api_schema = create_app(RetrieverStub()).openapi()
    search_operation = api_schema["paths"]["/search"]["post"]
    assert "requestBody" in search_operation
    assert not any(
        parameter.get("name") == "request"
        for parameter in search_operation.get("parameters", [])
    )

    paired = paired_delta(
        {"q": {"official_loose_accuracy": 1.0}},
        {"q": {"official_loose_accuracy": 0.5}},
        "official_loose_accuracy",
    )
    assert paired["paired_samples"] == 1
    assert paired["left_minus_right"] == 0.5

    assert parse_action("<think>x</think><search>alpha beta</search>", ["search"])
    parsed = parse_action("<answer>A long structured answer</answer>", ["answer"])
    assert parsed and parsed["payload"] == "A long structured answer"
    question = "List all requested entities and their attributes."
    assert "fan-out" in initial_search_prompt(question).casefold()
    assert "<search>" in decision_prompt(question, [{
        "search_round": 1, "query": "alpha",
        "evidence": {"evidence_text": "result:\ntext"}
    }], 2)
    assert "<answer>" in forced_answer_prompt(question, [{
        "search_round": 1, "query": "alpha",
        "evidence": {"evidence_text": "result:\ntext"}
    }])

    try:
        from fanoutqa_eval_local import answer_in_text, provenance
    except Exception:
        print("Official evaluator import skipped; run install_eval_deps.sh on server.")
    else:
        result = answer_in_text(["New York", "Boston"], "New York and Boston")
        assert result.found and result.score == 1
        source = provenance()
        assert not source["llm_judge_imported"]
        assert not source["openai_api_required"]
    print(
        "FANOUTQA WIKI2023 PROTOCOL SELF-TEST PASSED: natural K=0..5, "
        "100-token/FULL budget grid, zero-result observation, Blank, prompts, "
        "RSS/Atom parser, passage selection."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
