#!/usr/bin/env python3
"""Load FanOutQA's official local metrics without importing its LLM judge.

The upstream ``fanoutqa.eval`` package imports ``scorer`` from ``__init__``;
that module imports the OpenAI-backed judge and constructs its client during
module import.  This experiment needs only the deterministic local functions
from ``eval/string.py`` and ``eval/utils.py``.  Loading those two source files
under a synthetic package preserves their official implementation and relative
imports while deliberately avoiding ``eval/__init__.py`` and ``eval/llm.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict

import fanoutqa


def _load_source(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official FanOutQA metric source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_PACKAGE_ROOT = Path(fanoutqa.__file__).resolve().parent
_EVAL_ROOT = _PACKAGE_ROOT / "eval"
_UTILS_PATH = _EVAL_ROOT / "utils.py"
_STRING_PATH = _EVAL_ROOT / "string.py"

for _path in (_UTILS_PATH, _STRING_PATH):
    if not _path.is_file():
        raise ImportError(f"FanOutQA official metric file is missing: {_path}")

# Make relative imports such as ``from ..models`` work, without executing the
# upstream fanoutqa/eval/__init__.py file.
_eval_package = types.ModuleType("fanoutqa.eval")
_eval_package.__package__ = "fanoutqa.eval"
_eval_package.__path__ = [str(_EVAL_ROOT)]
sys.modules["fanoutqa.eval"] = _eval_package

_utils = _load_source("fanoutqa.eval.utils", _UTILS_PATH)
_string = _load_source("fanoutqa.eval.string", _STRING_PATH)

answer_in_text = _string.answer_in_text
str_answer = _utils.str_answer


def provenance() -> Dict[str, Any]:
    """Return auditable source paths/hashes and proof that no judge was loaded."""

    return {
        "loader": "direct_official_source_without_eval_package_initializer",
        "fanoutqa_version": getattr(fanoutqa, "__version__", None),
        "string_source": str(_STRING_PATH),
        "string_sha256": hashlib.sha256(_STRING_PATH.read_bytes()).hexdigest(),
        "utils_source": str(_UTILS_PATH),
        "utils_sha256": hashlib.sha256(_UTILS_PATH.read_bytes()).hexdigest(),
        "llm_judge_imported": "fanoutqa.eval.llm" in sys.modules,
        "openai_api_required": False,
    }

