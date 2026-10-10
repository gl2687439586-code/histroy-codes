#!/usr/bin/env bash
set -euo pipefail

if [[ -z "$(printenv PYTHON_BIN 2>/dev/null || true)" ]]; then
  PYTHON_BIN=/home/researcher/smartsearch_rebuild/envs/src-core/bin/python
fi
if [[ -z "$(printenv PIP_INDEX_URL 2>/dev/null || true)" ]]; then
  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
fi

FANOUTQA_SPEC="${FANOUTQA_SPEC:-fanoutqa[eval] @ git+https://github.com/zhudotexe/fanoutqa.git@main}"

if ! "$PYTHON_BIN" -m pip install --upgrade \
  --index-url "$PIP_INDEX_URL" \
  "$FANOUTQA_SPEC" "rouge-score>=0.1.2,<0.2" \
  "spacy>=3.7,<3.8" "ftfy>=6,<7"; then
  echo "Domestic PyPI mirror failed; retrying official PyPI." >&2
  "$PYTHON_BIN" -m pip install --upgrade \
    "$FANOUTQA_SPEC" "rouge-score>=0.1.2,<0.2" \
    "spacy>=3.7,<3.8" "ftfy>=6,<7"
fi

if ! "$PYTHON_BIN" -c 'import en_core_web_sm' >/dev/null 2>&1; then
  if ! "$PYTHON_BIN" -m pip install \
    "https://ghfast.top/https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"; then
    "$PYTHON_BIN" -m pip install \
      "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$SCRIPT_DIR" "$PYTHON_BIN" - <<'PY'
import en_core_web_sm
import fanoutqa
import os
import sys
sys.path.insert(0, os.environ["SCRIPT_DIR"])
from fanoutqa_eval_local import answer_in_text, provenance, str_answer
from rouge_score.rouge_scorer import RougeScorer
source = provenance()
assert not source["llm_judge_imported"], source
assert not source["openai_api_required"], source
print("FanOutQA official evaluation dependencies ready.")
print(answer_in_text(["New York", "Boston"], "New York and Boston"))
print(source)
PY
