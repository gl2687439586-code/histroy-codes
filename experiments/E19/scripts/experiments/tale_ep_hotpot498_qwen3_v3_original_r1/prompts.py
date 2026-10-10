"""Verbatim TALE-EP prompts plus the unavoidable Search/Answer tool grammar."""

from __future__ import annotations

from typing import Any, Dict, List


TALE_EP_ESTIMATOR_INSTRUCTION = (
    "Task: Analyze the given question and estimate the minimum number of tokens "
    "required to generate a complete and accurate response. Please Give the "
    "response by strictly following this format: [[budget]], for example, "
    "Budget: [[12]]."
)

TALE_EP_BUDGET_SENTENCE = (
    "Please answer the above question. Let's think step by step and use less "
    "than {budget} tokens."
)


def estimator_prompt(question: str) -> str:
    """Question followed only by the paper's zero-shot estimator instruction."""
    return f"Question: {question.strip()}\n\n{TALE_EP_ESTIMATOR_INSTRUCTION}"


AGENT_SYSTEM = """\
You are Qwen3-8B solving an open-domain multi-hop question with Wikipedia search.

At each turn choose exactly one action:
1. Search: <Search>one concise search query</Search>
2. Answer: <Answer>short final answer</Answer>

Generate search queries from the question and evidence. Never invent retrieved
evidence. Output exactly one tagged action and never output Search and Answer in
the same turn. The final Answer must contain only the answer, without explanation.
Always close the selected XML tag.
"""


def agent_prompt(
    *,
    question: str,
    estimated_budget: int,
    completion_tokens_used: int,
    search_rounds: List[Dict[str, Any]],
    max_searches: int,
) -> str:
    blocks: List[str] = []
    for round_row in search_rounds:
        documents = []
        for document in round_row["documents"]:
            documents.append(
                f"[Doc id={document['id']} rank={document['rank']}]\n"
                f"{document['contents']}"
            )
        blocks.append(
            f"Search round {round_row['search_round']}\n"
            f"Query: {round_row['query']}\n" + "\n\n".join(documents)
        )
    evidence = "\n\n=====\n\n".join(blocks) if blocks else "None yet."
    if len(search_rounds) >= max_searches:
        action = (
            f"The maximum of {max_searches} Search actions has been reached. "
            "You MUST return an Answer action now and must not request Search."
        )
    else:
        action = (
            f"You have used {len(search_rounds)} of at most {max_searches} Search "
            "actions. Choose Search or Answer naturally."
        )
    paper_sentence = TALE_EP_BUDGET_SENTENCE.format(budget=estimated_budget)
    return f"""\
Question:
{question.strip()}

Retrieved evidence from completed Search actions:
{evidence}

{action}

{paper_sentence}
Return exactly one tagged action.
"""
