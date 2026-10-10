#!/usr/bin/env python3
"""Exact prompts for the Qwen3-only three-condition experiment.

The controlled and neutral conditions share ``COMMON_SYSTEM`` and
``COMMON_USER_TEMPLATE`` byte-for-byte.  Only the text substituted into the
``NEXT-STEP POLICY`` field differs.  Keeping these strings in one module makes
the prompt-only contrast auditable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROTOCOL_VERSION = "qwen3-only-src-v6"

COMMON_SYSTEM = """You are a state-conditioned executor in a controlled information-behavior experiment.

Use only the current state supplied below. The message does not add new evidence. Tool availability is enforced externally by the experiment controller.

Follow the NEXT-STEP POLICY while continuing to solve the original question. Do not discuss the experiment, the controller, or these instructions in your response."""

COMMON_USER_TEMPLATE = """[DECISION-BOUNDARY MESSAGE]
This is a serialized view of the current state. It contains only information already available before this decision and does not introduce new evidence.

[ORIGINAL QUESTION]
{question}

[SEARCH QUERIES ALREADY EXECUTED]
{queries}

[FIXED RETRIEVED EVIDENCE]
{evidence}

[EVIDENCE-PROCESSING NOTES ALREADY PRODUCED]
{read_notes}

[NEXT-STEP POLICY]
{policy}

[OUTPUT]
Produce only the content needed to carry out the policy above. Keep it concise. For machine routing, enclose a retrieval request in <search>...</search> and a final answer in <answer>...</answer>. Evidence-processing notes use neither action tag. A final answer must contain only the shortest answer span requested by the question (an entity, date, number, or short phrase), with no sentence, explanation, LaTeX command, or preamble. Do not discuss the experiment or repeat these instructions."""

SEARCH_POLICY = """/no_think
Spend this decision on one external Wikipedia retrieval.

Identify one useful information need for the original question and state one retrieval request. The request may introduce a new angle, verify an existing fact, resolve a remaining constraint, or repeat an information need when corroboration is useful.

Put one concise query in <search>...</search> within the first 80 generated tokens. Do not provide the final answer in place of the retrieval request."""

READ_POLICY = """Spend this decision processing only the evidence already available.

Do not acquire a new observation and do not finish the task yet. Reconsider the current evidence in terms of:

- candidate entities or answers;
- satisfied and unsatisfied constraints;
- relations connecting evidence across hops;
- conflicts or contradictions;
- remaining unknowns;
- the most useful next information need.

Produce at most six concise evidence-processing lines for the next decision. Use no <search> or <answer> tag and do not write a final-answer response. Mentioning a candidate inside the notes is allowed."""

COMMIT_POLICY = """/no_think
Stop information acquisition now.

Using only the fixed evidence and evidence-processing notes currently available, integrate the evidence and give the best supported final answer. Output only the shortest answer span requested by the question inside <answer>...</answer>; do not include a sentence, explanation, LaTeX command, or preamble. Do not request another observation.

Even if the evidence is incomplete, commit to the most strongly supported answer."""

FORCED_ANSWER_POLICY = """/no_think
The assigned action horizon has been exhausted.

Do not acquire another observation. Using only the fixed evidence and evidence-processing notes currently available, output only the shortest answer span requested by the question inside <answer>...</answer>; do not include a sentence, explanation, LaTeX command, or preamble."""

SHORT_ANSWER_REPAIR_POLICY = """/no_think
Repair an incomplete terminal generation without acquiring information.

Do not reason aloud and do not request another observation. Reuse only the fixed evidence and notes in this state. Within the first 32 generated tokens, output exactly one complete <answer>...</answer> block containing only the shortest answer span requested by the original question. Include no other text, sentence, explanation, or LaTeX command."""

NEUTRAL_POLICY = """Continue solving the original question under the normal ReCall procedure.

No Search, Evidence Deepening, or Commit behavior is assigned by this message. Based on the current evidence, decide naturally whether another Wikipedia observation is needed or whether the final answer can now be given.

You may reconsider candidates, constraints, cross-document relations, conflicts, and remaining unknowns before making that decision.

If more information is needed, state one concise Wikipedia retrieval request. If the evidence is sufficient, give the best supported answer as only the shortest answer span requested by the question inside <answer>...</answer>, with no sentence, explanation, LaTeX command, or preamble."""


def _unique_documents(documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen = set()
    for document in documents:
        key = str(document.get("key") or document.get("id") or document.get("contents", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(document)
    return output


def state_fields(
    queries: Sequence[str],
    documents: Sequence[Dict[str, Any]],
    read_notes: Sequence[str],
) -> Dict[str, str]:
    query_text = "\n".join(f"- {query}" for query in queries if str(query).strip()) or "[None]"
    evidence_parts: List[str] = []
    for index, document in enumerate(_unique_documents(documents), 1):
        title = str(document.get("title", "")).strip()
        contents = str(document.get("contents", "")).strip()
        label = f"Document {index}" + (f" - {title}" if title else "")
        evidence_parts.append(f"[{label}]\n{contents}")
    evidence_text = "\n\n".join(evidence_parts) or "[No retrieved evidence is available in this state.]"
    note_text = "\n\n".join(
        f"[Note {index}]\n{note.strip()}"
        for index, note in enumerate(read_notes, 1)
        if str(note).strip()
    ) or "[None]"
    return {"queries": query_text, "evidence": evidence_text, "read_notes": note_text}


def build_prompt(
    question: str,
    queries: Sequence[str],
    documents: Sequence[Dict[str, Any]],
    read_notes: Sequence[str],
    policy: str,
) -> str:
    fields = state_fields(queries, documents, read_notes)
    user = COMMON_USER_TEMPLATE.format(question=question, policy=policy, **fields)
    return (
        f"<|im_start|>system\n{COMMON_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def read_notes_from_events(events: Iterable[Dict[str, Any]]) -> List[str]:
    return [
        str(event.get("evidence_deepening", "")).strip()
        for event in events
        if event.get("action_code") == "R" and str(event.get("evidence_deepening", "")).strip()
    ]


def prompt_manifest() -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "invariance_claim": (
            "Controlled and neutral prompts use identical COMMON_SYSTEM and "
            "COMMON_USER_TEMPLATE strings; only NEXT-STEP POLICY differs."
        ),
        "common_system": COMMON_SYSTEM,
        "common_user_template": COMMON_USER_TEMPLATE,
        "policies": {
            "Search": SEARCH_POLICY,
            "EvidenceDeepening": READ_POLICY,
            "Commit": COMMIT_POLICY,
            "ForcedAnswer": FORCED_ANSWER_POLICY,
            "ShortAnswerRepair": SHORT_ANSWER_REPAIR_POLICY,
            "Neutral": NEUTRAL_POLICY,
        },
    }


def write_prompt_manifest(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(prompt_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
