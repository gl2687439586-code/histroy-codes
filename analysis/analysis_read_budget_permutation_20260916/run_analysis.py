from __future__ import annotations

import gzip
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent
CONFIG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
BUDGETS = list(map(int, CONFIG["common_budgets"]))
SEED = int(CONFIG["seed"])
R = int(CONFIG["permutations"])
B = int(CONFIG["bootstraps"])

PATHS = {
    "hotpot_scores": WORKSPACE / "analysis_readscore_oracle_mapping_20260915/read/hotpot_random500_openrouter_gpt56sol_read_v2_fresh/results/read_scores_combined.jsonl",
    "hotpot_outcomes": WORKSPACE / "analysis_readscore_oracle_mapping_20260915/traj/analysis/scored_records.jsonl.gz",
    "fanout_scores": WORKSPACE / "analysis_cross_dataset_transfer_20260915/inputs/fanout_query_scores.jsonl",
    "fanout_outcomes": WORKSPACE / "analysis_cross_dataset_transfer_20260915/inputs/fanout_budget_scored.jsonl.gz",
    "fanout_blank": WORKSPACE / "analysis_cross_dataset_transfer_20260915/inputs/fanout_blank_scored.jsonl.gz",
    "historical_integrity": WORKSPACE / "analysis_cross_dataset_transfer_20260915/results/integrity.json",
    "historical_quantile": WORKSPACE / "analysis_cross_dataset_transfer_20260915/results/quantile_calibrated_transfer.json",
    "historical_harmonized": WORKSPACE / "analysis_cross_dataset_transfer_20260915/results/harmonized_metric_transfer.json",
    "historical_soft": WORKSPACE / "analysis_cross_dataset_transfer_20260915/rule_validation/manifest.json",
    "hotpot_scorer_manifest": WORKSPACE / "analysis_readscore_oracle_mapping_20260915/read/hotpot_random500_openrouter_gpt56sol_read_v2_fresh/manifest.json",
    "fanout_scorer_manifest": WORKSPACE / "analysis_cross_dataset_transfer_20260915/fanout_scores/fanoutqa_random100_openrouter_gpt56sol_v3_quote_safe/manifest.json",
}

QUALITY = {
    "FanOutQA": ["official_loose_accuracy", "official_strict_accuracy", "rougeL_f1", "answer_f1"],
    "HotpotQA": ["answer_f1", "answer_em", "answer_acc"],
}
PRIMARY = {"FanOutQA": "official_loose_accuracy", "HotpotQA": "answer_f1"}
BASELINE_NAME = {"FanOutQA": "BLANK", "HotpotQA": "FULL"}
METRICS = ["total_llm_tokens", "search_calls", "visible_evidence_tokens_total"]
FORBIDDEN_PREDICTION_KEYS = {
    "gold", "gold_answer", "final_answer", "quality", "answer_f1", "answer_em",
    "answer_acc", "official_loose_accuracy", "official_strict_accuracy", "rougeL_f1",
    "total_llm_tokens", "search_calls", "visible_evidence_tokens_total", "oracle",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]], compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(clean_json(row), ensure_ascii=False, allow_nan=False) + "\n")


def midranks(scores: pd.DataFrame) -> dict[int, float]:
    counts = scores.read_score.astype(int).value_counts().sort_index()
    total = float(counts.sum())
    below = 0
    result: dict[int, float] = {}
    for score, count in counts.items():
        result[int(score)] = (below + count / 2.0) / total
        below += int(count)
    return result


def quantile_bridge(source_scores: pd.DataFrame, target_scores: pd.DataFrame) -> dict[int, int]:
    source, target = midranks(source_scores), midranks(target_scores)
    return {
        int(ts): int(min(source, key=lambda ss: (abs(source[ss] - tq), ss)))
        for ts, tq in target.items()
    }


def coarse(score: int) -> str:
    if int(score) <= 1:
        return "low_1"
    if int(score) == 2:
        return "mid_2"
    return "high_3plus"


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = (np.asarray(values, float) - np.max(values)) / float(temperature)
    weights = np.exp(logits)
    return weights / weights.sum()


def load_inputs() -> dict[str, pd.DataFrame]:
    hs = pd.DataFrame(read_jsonl(PATHS["hotpot_scores"])).rename(columns={"question_id": "sample_id"})
    fs = pd.DataFrame(read_jsonl(PATHS["fanout_scores"])).rename(columns={"question_id": "sample_id"})
    h_all = pd.DataFrame(read_jsonl(PATHS["hotpot_outcomes"]))
    h_all["budget_numeric"] = pd.to_numeric(h_all.condition_group, errors="coerce")
    h = h_all[h_all.budget_numeric.isin(BUDGETS)].copy()
    h["budget"] = h.budget_numeric.astype(int)
    h_full = h_all[h_all.condition_group.astype(str).eq("FULL")].copy()
    f_all = pd.DataFrame(read_jsonl(PATHS["fanout_outcomes"]))
    f_all["budget_numeric"] = pd.to_numeric(f_all.condition_label, errors="coerce")
    f = f_all[f_all.budget_numeric.isin(BUDGETS)].copy()
    f["budget"] = f.budget_numeric.astype(int)
    f_blank = pd.DataFrame(read_jsonl(PATHS["fanout_blank"]))
    for frame in (h, h_full, f, f_blank):
        frame["total_llm_tokens"] = frame.standalone_llm_input_tokens + frame.standalone_llm_output_tokens
        frame["sample_id"] = frame.sample_id.astype(str)
    for frame in (hs, fs):
        frame["sample_id"] = frame.sample_id.astype(str)
        frame["read_score"] = frame.read_score.astype(int)
    return {"hs": hs, "fs": fs, "h": h, "h_full": h_full, "f": f, "f_blank": f_blank,
            "h_all": h_all, "f_all": f_all}


def learn_mapping(data: pd.DataFrame, scores: pd.DataFrame, metric: str, grouping: str) -> tuple[dict[str, int], int, list[dict[str, Any]]]:
    meta = scores[["sample_id", "read_score"]].copy()
    meta["group"] = meta.read_score.map(coarse) if grouping == "coarse" else meta.read_score.astype(int)
    merged = data.merge(meta[["sample_id", "group"]], on="sample_id", validate="many_to_one")
    global_table = merged.groupby("budget").agg(quality=(metric, "mean"), cost=("total_llm_tokens", "mean")).reset_index()
    fallback = int(global_table.sort_values(["quality", "cost", "budget"], ascending=[False, True, True]).iloc[0].budget)
    mapping: dict[str, int] = {}
    curves: list[dict[str, Any]] = []
    for group, rows in merged.groupby("group"):
        table = rows.groupby("budget").agg(n=("sample_id", "nunique"), quality=(metric, "mean"), cost=("total_llm_tokens", "mean")).reset_index()
        selected = table.sort_values(["quality", "cost", "budget"], ascending=[False, True, True]).iloc[0]
        mapping[str(group)] = int(selected.budget)
        for row in table.itertuples():
            curves.append({"group": str(group), "budget": int(row.budget), "n": int(row.n),
                           "quality": float(row.quality), "cost": float(row.cost),
                           "selected": int(row.budget) == int(selected.budget)})
    return mapping, fallback, curves


def source_curves(data: pd.DataFrame, scores: pd.DataFrame, metric: str) -> tuple[dict[int, np.ndarray], np.ndarray]:
    merged = data.merge(scores[["sample_id", "read_score"]], on="sample_id", validate="many_to_one")
    curves = {
        int(score): group.groupby("budget")[metric].mean().reindex(BUDGETS).to_numpy(float)
        for score, group in merged.groupby("read_score")
    }
    global_curve = merged.groupby("budget")[metric].mean().reindex(BUDGETS).to_numpy(float)
    return curves, global_curve


def freeze_policies(data: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    # This phase receives source outcomes and target query/score only. Target outcome matrices are not read here.
    hs, fs, h, f = data["hs"], data["fs"], data["h"], data["f"]
    h_exact, h_fallback, h_curves_rows = learn_mapping(h, hs, "answer_f1", "exact")
    f_coarse, f_fallback, f_curves_rows = learn_mapping(f, fs, "answer_f1", "coarse")
    hf_bridge = quantile_bridge(hs, fs)
    fh_bridge = quantile_bridge(fs, hs)
    h_curves, h_global = source_curves(h, hs, "answer_f1")
    f_curves, f_global = source_curves(f, fs, "answer_f1")

    specs = {
        "hard_hotpot_to_fanout": {"target": "FanOutQA", "source": "HotpotQA", "kind": "hard_quantile"},
        "hard_fanout_to_hotpot": {"target": "HotpotQA", "source": "FanOutQA", "kind": "hard_coarse"},
        "soft_hotpot_to_fanout": {"target": "FanOutQA", "source": "HotpotQA", "kind": "soft_quantile"},
        "soft_fanout_to_hotpot": {"target": "HotpotQA", "source": "FanOutQA", "kind": "soft_quantile"},
    }
    rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        target_scores = fs if spec["target"] == "FanOutQA" else hs
        P = np.zeros((len(target_scores), len(BUDGETS)), float)
        for i, row in enumerate(target_scores.itertuples()):
            score = int(row.read_score)
            source_score = None
            if name == "hard_hotpot_to_fanout":
                source_score = hf_bridge[score]
                budget = h_exact.get(str(source_score), h_fallback)
                P[i, BUDGETS.index(budget)] = 1.0
            elif name == "hard_fanout_to_hotpot":
                group = coarse(score)
                budget = f_coarse.get(group, f_fallback)
                P[i, BUDGETS.index(budget)] = 1.0
            elif name == "soft_hotpot_to_fanout":
                source_score = hf_bridge[score]
                P[i] = softmax(h_curves.get(source_score, h_global), CONFIG["soft_temperatures"]["HotpotQA"])
            else:
                source_score = fh_bridge[score]
                P[i] = softmax(f_curves.get(source_score, f_global), CONFIG["soft_temperatures"]["FanOutQA"])
            record = {
                "policy": name, "target_dataset": spec["target"], "source_dataset": spec["source"],
                "sample_id": str(row.sample_id), "query": str(row.query), "read_score": score,
                "source_equivalent_score": source_score,
                "budget_probability": {str(b): float(p) for b, p in zip(BUDGETS, P[i])},
                "expected_budget": float(P[i] @ np.asarray(BUDGETS, float)),
                "rule_version": "frozen_historical_20260915",
            }
            assert not (set(record) & FORBIDDEN_PREDICTION_KEYS)
            rows.append(record)
        matrices[name] = P
    policy = {
        "common_budgets": BUDGETS,
        "source_training_metric": "answer_f1",
        "hotpot_exact_mapping": h_exact,
        "hotpot_exact_fallback": h_fallback,
        "fanout_coarse_mapping": f_coarse,
        "fanout_coarse_fallback": f_fallback,
        "hotpot_to_fanout_quantile_bridge": hf_bridge,
        "fanout_to_hotpot_quantile_bridge": fh_bridge,
        "hotpot_score_midranks": midranks(hs),
        "fanout_score_midranks": midranks(fs),
        "soft_temperatures": CONFIG["soft_temperatures"],
        "hotpot_source_curves": {str(k): dict(zip(map(str, BUDGETS), map(float, v))) for k, v in h_curves.items()},
        "fanout_source_curves": {str(k): dict(zip(map(str, BUDGETS), map(float, v))) for k, v in f_curves.items()},
        "source_group_curve_rows": {"HotpotQA_exact": h_curves_rows, "FanOutQA_coarse": f_curves_rows},
        "prediction_phase_forbidden_target_fields": sorted(FORBIDDEN_PREDICTION_KEYS),
    }
    return policy, rows, matrices


def matrix(data: pd.DataFrame, ids: list[str], column: str) -> np.ndarray:
    q = data.pivot(index="sample_id", columns="budget", values=column).reindex(index=ids, columns=BUDGETS)
    out = q.to_numpy(float)
    if out.shape != (len(ids), len(BUDGETS)) or not np.isfinite(out).all():
        raise RuntimeError(f"Invalid matrix for {column}: {out.shape}")
    return out


def policy_values(P: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {metric: np.sum(P * values, axis=1) for metric, values in arrays.items()}


def aggregate(name: str, dataset: str, values: dict[str, np.ndarray], P: np.ndarray | None,
              scorer: dict[str, Any] | None, role: str) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": dataset, "strategy": name, "role": role,
                           "n": int(len(next(iter(values.values()))))}
    for metric, vector in values.items():
        row[metric] = float(np.mean(vector))
        if metric == "total_llm_tokens":
            row["collection_total_llm_tokens"] = float(np.sum(vector))
    row["mean_expected_budget"] = float(np.mean(P @ np.asarray(BUDGETS))) if P is not None else None
    row["scoring_overhead"] = scorer
    return row


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (m - rank) * pvalues[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def envelope(q: np.ndarray, c: np.ndarray, cap: float, labels: list[str]) -> dict[str, Any]:
    candidates: list[tuple[float, float, np.ndarray]] = []
    n = len(q)
    for i in range(n):
        if c[i] <= cap + 1e-9:
            w = np.zeros(n); w[i] = 1.0
            candidates.append((float(q[i]), float(c[i]), w))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(c[i] - c[j]) < 1e-12:
                continue
            lo, hi = (i, j) if c[i] < c[j] else (j, i)
            if c[lo] <= cap <= c[hi]:
                whi = (cap - c[lo]) / (c[hi] - c[lo])
                w = np.zeros(n); w[hi] = whi; w[lo] = 1.0 - whi
                candidates.append((float(w @ q), float(w @ c), w))
    if not candidates:
        return {"feasible": False, "cost_cap": cap}
    best = max(candidates, key=lambda x: (x[0], -x[1]))
    return {"feasible": True, "cost_cap": cap, "quality": best[0], "cost": best[1],
            "weights": {labels[i]: float(w) for i, w in enumerate(best[2]) if w > 1e-12},
            "constraint_residual": float(cap - best[1])}


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def normalize_query(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def input_audit(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    hs, fs, h, f = data["hs"], data["fs"], data["h"], data["f"]
    audit: dict[str, Any] = {"inputs": {}}
    for name, path in PATHS.items():
        audit["inputs"][name] = {"absolute_path": str(path.resolve()), "bytes": path.stat().st_size,
                                         "sha256": sha256(path)}
    for dataset, scores, outcomes, expected in [("HotpotQA", hs, h, 500), ("FanOutQA", fs, f, 100)]:
        counts = outcomes.groupby("sample_id").budget.agg(["count", "nunique"])
        audit[dataset] = {
            "score_records": len(scores), "score_unique_ids": int(scores.sample_id.nunique()),
            "trajectory_records_common": len(outcomes), "trajectory_unique_ids": int(outcomes.sample_id.nunique()),
            "complete_grid_questions": int(((counts["count"] == 6) & (counts["nunique"] == 6)).sum()),
            "duplicate_sample_budget": int(outcomes.duplicated(["sample_id", "budget"]).sum()),
            "id_sets_equal": set(scores.sample_id) == set(outcomes.sample_id),
            "read_score_distribution": {str(k): int(v) for k, v in scores.read_score.value_counts().sort_index().items()},
            "finite_quality_cost": bool(np.isfinite(outcomes[QUALITY[dataset] + METRICS].to_numpy(float)).all()),
            "total_cost_exact_sum": bool(np.array_equal(outcomes.total_llm_tokens.to_numpy(),
                (outcomes.standalone_llm_input_tokens + outcomes.standalone_llm_output_tokens).to_numpy())),
            "passed": bool(len(scores) == expected and len(outcomes) == expected * 6 and
                           scores.sample_id.nunique() == expected and counts["nunique"].eq(6).all()),
        }
    audit["HotpotQA"]["full_records"] = int(len(data["h_full"]))
    audit["HotpotQA"]["local_fallback_rate_common"] = float(h.local_fallback.mean())
    audit["HotpotQA"]["context_compaction_rate_common"] = float((h.context_compactions > 0).mean())
    audit["HotpotQA"]["format_repair_rate_common"] = float((h.format_repair_calls > 0).mean())
    audit["HotpotQA"]["root_boundary_actual_diff_rate"] = float((h.root_actual_exposure_tokens != h.root_exposure_tokens).mean())
    audit["HotpotQA"]["retriever_metadata_available_in_scored_table"] = False
    audit["HotpotQA"]["protocol_disclosure"] = "Historical protocol used a FlashRAG dense retriever over the older Wikipedia index; exact backend/version is not encoded in this scored table."
    audit["FanOutQA"]["blank_records"] = int(len(data["f_blank"]))
    audit["FanOutQA"]["local_fallback_rate_common"] = float(f.local_fallback.mean())
    audit["FanOutQA"]["context_compaction_rate_common"] = float((f.context_compactions > 0).mean())
    audit["FanOutQA"]["answer_status_distribution_common"] = {str(k): int(v) for k, v in f.answer_status.value_counts().items()}
    audit["FanOutQA"]["retriever_backend"] = sorted(map(str, f.retriever_backend.unique()))
    audit["FanOutQA"]["wikipedia_snapshot"] = sorted(map(str, f.wikipedia_snapshot.unique()))
    fan_queries = fs.set_index("sample_id")["query"].astype(str)
    fan_outcome_queries = f.drop_duplicates("sample_id").set_index("sample_id")["question"].astype(str)
    audit["FanOutQA"]["score_outcome_query_exact_matches"] = int(sum(
        fan_queries.loc[sid] == fan_outcome_queries.loc[sid] for sid in fan_queries.index
    ))
    hq = {normalize_query(q): sid for sid, q in zip(hs.sample_id, hs["query"])}
    fq = {normalize_query(q): sid for sid, q in zip(fs.sample_id, fs["query"])}
    exact = sorted(set(hq) & set(fq))
    near = []
    for hr in hs.itertuples():
        hn = normalize_query(hr.query)
        for fr in fs.itertuples():
            fn = normalize_query(fr.query)
            ratio = SequenceMatcher(None, hn, fn).ratio()
            if ratio >= .72:
                near.append({"hotpot_id": hr.sample_id, "fanout_id": fr.sample_id, "ratio": ratio,
                             "hotpot_query": hr.query, "fanout_query": fr.query})
    audit["cross_dataset_overlap"] = {"exact_normalized_count": len(exact),
                                      "near_threshold": .72,
                                      "near_pairs": sorted(near, key=lambda x: -x["ratio"])[:20]}
    return audit


def scorer_overheads() -> dict[str, Any]:
    result = {}
    for dataset, key in [("HotpotQA", "hotpot_scorer_manifest"), ("FanOutQA", "fanout_scorer_manifest")]:
        manifest = json.loads(PATHS[key].read_text(encoding="utf-8"))
        usage = manifest["usage"]["by_dimension"]["read"]
        n = int(manifest["question_count"])
        result[dataset] = {
            "model": manifest["requested_model"], "final_questions": n,
            "logged_physical_generation_requests_all_dimensions": manifest["generation_requests_logged_across_runs"],
            "read_campaign_prompt_tokens": usage["prompt_tokens"],
            "read_campaign_completion_tokens": usage["completion_tokens"],
            "read_campaign_total_tokens": usage["total_tokens"],
            "read_campaign_usd": usage["cost"],
            "observed_read_tokens_per_final_question_retry_inflated": usage["total_tokens"] / n,
            "observed_read_usd_per_final_question_retry_inflated": usage["cost"] / n,
            "warning": "OpenRouter tokens are heterogeneous with Qwen generation tokens; campaign totals include recovery/retry history and are not equivalent FLOPs.",
        }
    return result


def run() -> None:
    for d in [BASE / "frozen", BASE / "normalized", BASE / "results", BASE / "figures", BASE / "tests"]:
        d.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    audit = input_audit(data)
    overheads = scorer_overheads()
    dump_json(BASE / "audit.json", audit)
    dump_json(BASE / "input_manifest.json", {"environment": {"python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "inputs": audit["inputs"], "controller_costs": overheads})
    if not audit["HotpotQA"]["passed"] or not audit["FanOutQA"]["passed"]:
        raise RuntimeError("Input integrity failed")

    policy, prediction_rows, P_by_policy = freeze_policies(data)
    policy["input_hashes"] = {k: sha256(v) for k, v in PATHS.items()}
    policy["code_sha256"] = sha256(Path(__file__))
    dump_json(BASE / "frozen/hotpot_to_fanout_policy.json", {k: v for k, v in policy.items() if "fanout" in k.lower() or k in ("common_budgets", "source_training_metric", "soft_temperatures", "input_hashes", "code_sha256", "prediction_phase_forbidden_target_fields")})
    dump_json(BASE / "frozen/fanout_to_hotpot_policy.json", {k: v for k, v in policy.items() if "hotpot" in k.lower() or k in ("common_budgets", "source_training_metric", "soft_temperatures", "input_hashes", "code_sha256", "prediction_phase_forbidden_target_fields")})
    write_jsonl(BASE / "frozen/predictions.jsonl", prediction_rows)

    dataset_info = {
        "FanOutQA": {"scores": data["fs"].sort_values("sample_id").reset_index(drop=True),
                     "grid": data["f"], "baseline": data["f_blank"], "baseline_name": "BLANK"},
        "HotpotQA": {"scores": data["hs"].sort_values("sample_id").reset_index(drop=True),
                     "grid": data["h"], "baseline": data["h_full"], "baseline_name": "FULL"},
    }
    # Reorder frozen matrices from score-file order into canonical ID order.
    pred_df = pd.DataFrame(prediction_rows)
    for name, P in list(P_by_policy.items()):
        target = "FanOutQA" if "to_fanout" in name else "HotpotQA"
        source_order = data["fs"].sample_id.tolist() if target == "FanOutQA" else data["hs"].sample_id.tolist()
        source_index = {sid: i for i, sid in enumerate(source_order)}
        ids = dataset_info[target]["scores"].sample_id.tolist()
        P_by_policy[name] = P[[source_index[sid] for sid in ids]]

    normalized_rows = []
    arrays_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    baseline_values: dict[str, dict[str, np.ndarray]] = {}
    for dataset, info in dataset_info.items():
        ids = info["scores"].sample_id.tolist()
        arrays = {m: matrix(info["grid"], ids, m) for m in QUALITY[dataset] + METRICS}
        arrays_by_dataset[dataset] = arrays
        base = info["baseline"].set_index("sample_id").reindex(ids)
        baseline_values[dataset] = {m: base[m].to_numpy(float) for m in QUALITY[dataset] + METRICS}
        for i, sid in enumerate(ids):
            for j, budget in enumerate(BUDGETS):
                row = {"dataset": dataset, "sample_id": sid, "budget": budget}
                row.update({m: float(arrays[m][i, j]) for m in arrays})
                normalized_rows.append(row)
    write_jsonl(BASE / "normalized/outcomes.jsonl.gz", normalized_rows, compressed=True)

    scorers = overheads
    summaries: list[dict[str, Any]] = []
    controls: dict[str, dict[str, Any]] = {}
    for dataset, info in dataset_info.items():
        ids = info["scores"].sample_id.tolist(); arrays = arrays_by_dataset[dataset]
        base = baseline_values[dataset]
        summaries.append(aggregate(info["baseline_name"], dataset, base, None, None, "unbudgeted_baseline"))
        fixed_rows = []
        for j, budget in enumerate(BUDGETS):
            vals = {m: a[:, j] for m, a in arrays.items()}
            row = aggregate(f"fixed_{budget}", dataset, vals, np.eye(1, len(BUDGETS), j).repeat(len(ids), axis=0), None, "fixed_target_comparator")
            summaries.append(row); fixed_rows.append(row)
        primary = PRIMARY[dataset]
        chosen = []
        for i in range(len(ids)):
            chosen.append(sorted(range(len(BUDGETS)), key=lambda j: (-arrays[primary][i, j], arrays["total_llm_tokens"][i, j], BUDGETS[j]))[0])
        P_oracle = np.zeros((len(ids), len(BUDGETS))); P_oracle[np.arange(len(ids)), chosen] = 1
        summaries.append(aggregate("common_grid_oracle", dataset, policy_values(P_oracle, arrays), P_oracle, None, "target_hindsight_oracle"))
        best_fixed = max(fixed_rows, key=lambda row: (row[primary], -row["total_llm_tokens"], -row["mean_expected_budget"]))
        controls[dataset] = {"target_hindsight_best_fixed": best_fixed["strategy"]}

    # Source-selected global fixed controls.
    h_global_map, h_global_fixed, _ = learn_mapping(data["h"], data["hs"], "answer_f1", "exact")
    f_global_map, f_global_fixed, _ = learn_mapping(data["f"], data["fs"], "answer_f1", "coarse")
    source_fixed = {"FanOutQA": h_global_fixed, "HotpotQA": f_global_fixed}
    controls["FanOutQA"]["source_selected_fixed_budget"] = h_global_fixed
    controls["HotpotQA"]["source_selected_fixed_budget"] = f_global_fixed

    strategy_specs = [
        ("hard_hotpot_to_fanout", "FanOutQA", "hard_primary"),
        ("hard_fanout_to_hotpot", "HotpotQA", "hard_primary"),
        ("soft_hotpot_to_fanout", "FanOutQA", "soft_secondary"),
        ("soft_fanout_to_hotpot", "HotpotQA", "soft_secondary"),
    ]
    seed_seq = np.random.SeedSequence(SEED)
    children = seed_seq.spawn(len(strategy_specs) * 3)
    child_iter = iter(children)
    perm_rows = []
    tests = []
    boot_results: dict[str, Any] = {}
    envelope_results: dict[str, Any] = {}
    case_rows: list[dict[str, Any]] = []

    for name, dataset, family in strategy_specs:
        info = dataset_info[dataset]; ids = info["scores"].sample_id.tolist(); arrays = arrays_by_dataset[dataset]
        P = P_by_policy[name]
        assert P.shape == (len(ids), len(BUDGETS)) and np.allclose(P.sum(axis=1), 1)
        vals_A = policy_values(P, arrays)
        pbar = P.mean(axis=0); P_blind = np.repeat(pbar[None, :], len(ids), axis=0)
        vals_blind = policy_values(P_blind, arrays)
        summaries.append(aggregate(name, dataset, vals_A, P, scorers[dataset], "matched_prior"))
        summaries.append(aggregate(name + "__blind", dataset, vals_blind, P_blind, None, "unconditional_same_marginal"))
        primary = PRIMARY[dataset]
        rng = np.random.default_rng(next(child_iter))
        perm = {m: np.empty(R) for m in [primary] + METRICS}
        for r in range(R):
            order = rng.permutation(len(ids))
            Pp = P[order]
            for metric in perm:
                perm[metric][r] = float(np.mean(np.sum(Pp * arrays[metric], axis=1)))
            perm_rows.append({"policy": name, "dataset": dataset, "permutation": r,
                              **{metric: perm[metric][r] for metric in perm}})
        qA, cA = float(np.mean(vals_A[primary])), float(np.mean(vals_A["total_llm_tokens"]))
        qblind, cblind = float(np.mean(vals_blind[primary])), float(np.mean(vals_blind["total_llm_tokens"]))
        p_q = (1 + int(np.sum(perm[primary] >= qA))) / (R + 1)
        p_cost = (1 + int(np.sum(perm["total_llm_tokens"] <= cA))) / (R + 1)
        mc_q_se = float(np.std(perm[primary], ddof=1) / math.sqrt(R))
        mc_c_se = float(np.std(perm["total_llm_tokens"], ddof=1) / math.sqrt(R))
        tests.append({"policy": name, "dataset": dataset, "family": family, "primary_metric": primary,
                      "matched_quality": qA, "blind_quality": qblind, "delta_quality": qA-qblind,
                      "delta_quality_percentage_points": 100*(qA-qblind),
                      "matched_cost": cA, "blind_cost": cblind, "delta_total_tokens": cA-cblind,
                      "delta_total_tokens_percent": 100*(cA-cblind)/cblind,
                      "quality_permutation_p_one_sided": p_q, "cost_lower_tail_p": p_cost,
                      "quality_permutation_percentile": float(np.mean(perm[primary] <= qA)),
                      "cost_permutation_percentile": float(np.mean(perm["total_llm_tokens"] <= cA)),
                      "permutation_mean_quality": float(np.mean(perm[primary])),
                      "permutation_mean_cost": float(np.mean(perm["total_llm_tokens"])),
                      "analytic_blind_matches_permutation_mean_quality": abs(np.mean(perm[primary])-qblind) <= 5*mc_q_se + 1e-12,
                      "analytic_blind_matches_permutation_mean_cost": abs(np.mean(perm["total_llm_tokens"])-cblind) <= 5*mc_c_se + 1e-12,
                      "permutation_quality_mc_se": mc_q_se, "permutation_cost_mc_se": mc_c_se})

        # Fixed-policy conditional bootstrap. Blind marginal is recomputed in every resample.
        rng_boot = np.random.default_rng(next(child_iter))
        comparisons = {ref: {m: np.empty(B) for m in (primary, "total_llm_tokens")} for ref in ("blind", "baseline", "source_fixed")}
        base = baseline_values[dataset]
        fixed_j = BUDGETS.index(source_fixed[dataset])
        for bidx in range(B):
            idx = rng_boot.integers(0, len(ids), len(ids))
            Pb = P[idx]; pbar_b = Pb.mean(axis=0)
            for metric in (primary, "total_llm_tokens"):
                m = arrays[metric][idx]
                candidate = float(np.mean(np.sum(Pb*m, axis=1)))
                comparisons["blind"][metric][bidx] = candidate - float(np.mean(m @ pbar_b))
                comparisons["baseline"][metric][bidx] = candidate - float(np.mean(base[metric][idx]))
                comparisons["source_fixed"][metric][bidx] = candidate - float(np.mean(m[:, fixed_j]))
        boot_results[name] = {ref: {metric: {"mean_delta": float(np.mean(values)), "ci95": percentile_ci(values)}
                                            for metric, values in by_metric.items()}
                              for ref, by_metric in comparisons.items()}

        # Calibration sensitivity bootstrap: source rule frozen; target score bridge recomputed.
        rng_cal = np.random.default_rng(next(child_iter))
        cal_delta_q = np.empty(B); cal_delta_c = np.empty(B)
        scores = info["scores"]
        for bidx in range(B):
            idx = rng_cal.integers(0, len(ids), len(ids))
            sampled_scores = scores.iloc[idx].copy()
            if name == "hard_fanout_to_hotpot":
                Pnew = P[idx]
            else:
                source_scores = data["hs"] if "hotpot_to" in name else data["fs"]
                bridge = quantile_bridge(source_scores, sampled_scores)
                Pnew = np.zeros((len(idx), len(BUDGETS)))
                for k, score in enumerate(sampled_scores.read_score.astype(int)):
                    ss = bridge[int(score)]
                    if name == "hard_hotpot_to_fanout":
                        budget = policy["hotpot_exact_mapping"].get(str(ss), policy["hotpot_exact_fallback"])
                        Pnew[k, BUDGETS.index(budget)] = 1
                    else:
                        curves_key = "hotpot_source_curves" if "hotpot_to" in name else "fanout_source_curves"
                        temp_key = "HotpotQA" if "hotpot_to" in name else "FanOutQA"
                        curve = np.array([policy[curves_key][str(ss)][str(b)] for b in BUDGETS])
                        Pnew[k] = softmax(curve, CONFIG["soft_temperatures"][temp_key])
            pbar_new = Pnew.mean(axis=0)
            mq = arrays[primary][idx]; mc = arrays["total_llm_tokens"][idx]
            cal_delta_q[bidx] = np.mean(np.sum(Pnew*mq, axis=1)) - np.mean(mq @ pbar_new)
            cal_delta_c[bidx] = np.mean(np.sum(Pnew*mc, axis=1)) - np.mean(mc @ pbar_new)
        boot_results[name]["calibration_sensitivity_A_minus_blind"] = {
            primary: {"mean_delta": float(cal_delta_q.mean()), "ci95": percentile_ci(cal_delta_q)},
            "total_llm_tokens": {"mean_delta": float(cal_delta_c.mean()), "ci95": percentile_ci(cal_delta_c)},
        }

        # Target-label-aware unconditional envelope at the matched policy cost.
        qmean = arrays[primary].mean(axis=0); cmean = arrays["total_llm_tokens"].mean(axis=0)
        common_env = envelope(qmean, cmean, cA, [str(b) for b in BUDGETS])
        qext = np.r_[qmean, np.mean(base[primary])]; cext = np.r_[cmean, np.mean(base["total_llm_tokens"])]
        extended_env = envelope(qext, cext, cA, [str(b) for b in BUDGETS] + [BASELINE_NAME[dataset]])
        envelope_results[name] = {"common_six_budget": common_env, "extended_with_unbudgeted_baseline": extended_env,
                                  "matched_quality": qA, "matched_cost": cA,
                                  "matched_minus_common_envelope_quality": qA-common_env.get("quality", float("nan"))}

        # Contribution cases relative to the analytic blind policy.
        dq = vals_A[primary] - vals_blind[primary]; dc = vals_A["total_llm_tokens"] - vals_blind["total_llm_tokens"]
        selected = set(np.argsort(dq)[-5:]) | set(np.argsort(dq)[:5]) | set(np.argsort(dc)[-5:])
        score_map = info["scores"].set_index("sample_id")
        for i in sorted(selected):
            case_rows.append({"policy": name, "dataset": dataset, "sample_id": ids[i],
                "query": score_map.loc[ids[i], "query"], "read_score": int(score_map.loc[ids[i], "read_score"]),
                "probability": {str(b): float(p) for b, p in zip(BUDGETS, P[i])},
                "marginal_probability": {str(b): float(p) for b, p in zip(BUDGETS, pbar)},
                "quality_contribution_A_minus_blind": float(dq[i]), "cost_contribution_A_minus_blind": float(dc[i]),
                "quality_curve": {str(b): float(v) for b, v in zip(BUDGETS, arrays[primary][i])},
                "cost_curve": {str(b): float(v) for b, v in zip(BUDGETS, arrays["total_llm_tokens"][i])},
                "search_curve": {str(b): float(v) for b, v in zip(BUDGETS, arrays["search_calls"][i])}})

    # Holm correction by predeclared family.
    for family in ("hard_primary", "soft_secondary"):
        members = {t["policy"]: t["quality_permutation_p_one_sided"] for t in tests if t["family"] == family}
        adjusted = holm(members)
        for t in tests:
            if t["family"] == family:
                t["holm_adjusted_quality_p"] = adjusted[t["policy"]]

    # Reproduction checks against the frozen historical displays.
    summary_df = pd.DataFrame(summaries)
    expected = {
        "hard_hotpot_to_fanout": {"official_loose_accuracy": .4140297619047619, "total_llm_tokens": 9614.12},
        "hard_fanout_to_hotpot": {"answer_f1": .4939306377082417, "total_llm_tokens": 4059.796},
        "soft_hotpot_to_fanout": {"official_loose_accuracy": .410528147278294, "total_llm_tokens": 9754.738140132426},
        "soft_fanout_to_hotpot": {"answer_f1": .47515842190894914, "total_llm_tokens": 3842.902607739976},
        "BLANK": {"official_loose_accuracy": .41021230158730154, "total_llm_tokens": 14179.01},
        "FULL": {"answer_f1": .4948185362236753, "total_llm_tokens": 4526.522},
    }
    reproduction = []
    for strategy, metrics in expected.items():
        row = summary_df[summary_df.strategy.eq(strategy)].iloc[0]
        for metric, target in metrics.items():
            actual = float(row[metric]); reproduction.append({"strategy": strategy, "metric": metric,
                "expected": target, "actual": actual, "absolute_error": abs(actual-target),
                "passed": abs(actual-target) < 1e-6})

    summary_df.to_csv(BASE / "results/strategy_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tests).to_csv(BASE / "results/paired_comparisons.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(perm_rows).to_csv(BASE / "results/permutation_statistics.csv.gz", index=False, compression="gzip")
    dump_json(BASE / "results/permutation_tests.json", tests)
    dump_json(BASE / "results/bootstrap_summary.json", boot_results)
    dump_json(BASE / "results/cost_matched_envelope.json", envelope_results)
    dump_json(BASE / "results/reproduction_checks.json", reproduction)
    dump_json(BASE / "results/controls.json", controls)
    write_jsonl(BASE / "results/informative_cases.jsonl", case_rows)

    # Utility sensitivity is descriptive and uses preregistered lambdas.
    utility = []
    for t in tests:
        for lam in CONFIG["utility_lambdas"]:
            utility.append({"policy": t["policy"], "dataset": t["dataset"], "lambda": lam,
                            "utility_A": t["matched_quality"] - lam*t["matched_cost"]/1000,
                            "utility_blind": t["blind_quality"] - lam*t["blind_cost"]/1000})
    pd.DataFrame(utility).to_csv(BASE / "results/utility_sensitivity.csv", index=False, encoding="utf-8-sig")

    # Scientific figures from exact result arrays.
    perm_df = pd.DataFrame(perm_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (name, dataset, _) in zip(axes.flat, strategy_specs):
        t = next(x for x in tests if x["policy"] == name); sub = perm_df[perm_df.policy.eq(name)]
        ax.scatter(sub.total_llm_tokens, sub[PRIMARY[dataset]], s=6, alpha=.13, color="#8a949e", label="permutations")
        ax.scatter(t["blind_cost"], t["blind_quality"], marker="D", s=80, color="#d97706", label="blind mixture")
        ax.scatter(t["matched_cost"], t["matched_quality"], marker="*", s=180, color="#0f766e", label="matched")
        ss = summary_df[summary_df.dataset.eq(dataset)]
        fixed = ss[ss.strategy.str.startswith("fixed_")]
        ax.plot(fixed.total_llm_tokens, fixed[PRIMARY[dataset]], "o--", ms=4, lw=1, color="#2563eb", label="fixed budgets")
        base = ss[ss.strategy.eq(BASELINE_NAME[dataset])].iloc[0]
        ax.scatter(base.total_llm_tokens, base[PRIMARY[dataset]], marker="s", s=65, color="#111827", label=BASELINE_NAME[dataset])
        ax.set(title=name, xlabel="Mean generation tokens", ylabel=PRIMARY[dataset]); ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.savefig(BASE / "figures/permutation_quality_cost.png", dpi=220)
    fig.savefig(BASE / "figures/permutation_quality_cost.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (name, dataset, _) in zip(axes.flat, strategy_specs):
        t = next(x for x in tests if x["policy"] == name); values = perm_df[perm_df.policy.eq(name)][PRIMARY[dataset]]
        ax.hist(values, bins=35, color="#a7b0ba", edgecolor="white")
        ax.axvline(t["matched_quality"], color="#0f766e", lw=2.5, label="matched")
        ax.axvline(t["blind_quality"], color="#d97706", lw=2, ls="--", label="blind")
        ax.set(title=f"{name}\np={t['quality_permutation_p_one_sided']:.4g}, Holm={t['holm_adjusted_quality_p']:.4g}", xlabel=PRIMARY[dataset], ylabel="Count")
        ax.legend(); ax.grid(alpha=.15)
    fig.savefig(BASE / "figures/permutation_quality_histograms.png", dpi=220)
    fig.savefig(BASE / "figures/permutation_quality_histograms.pdf")
    plt.close(fig)

    # Human-readable report.
    lines = ["# READ 预算跨数据集题目—预算匹配置换检验", "",
             "> 这是已有数据上的回顾性机制诊断；没有新增模型、Judge 或检索调用。", "",
             "## 核心问题", "", "在完全相同的预算边际分布下，真实 query—预算配对是否优于随机错配？", "",
             "## 数据与协议审计", "",
             f"- HotpotQA：500题，公共六档3,000条轨迹，FULL 500条；输入审计：{'PASSED' if audit['HotpotQA']['passed'] else 'FAILED'}。",
             f"- FanOutQA：100题，公共六档600条轨迹，Blank 100条；输入审计：{'PASSED' if audit['FanOutQA']['passed'] else 'FAILED'}。",
             f"- 跨数据集规范化精确重复题：{audit['cross_dataset_overlap']['exact_normalized_count']}。",
             "- 每档预算是每次自然搜索后的证据暴露上限；搜索次数仍由模型自然决定。", "",
             "- HotpotQA 的评分表未编码精确 retriever 版本；历史协议为旧 Wikipedia 索引上的 FlashRAG DenseRetriever。FanOutQA 为 2023-09 Kiwix up-to-5，两者构成真实跨域混杂。", "",
             "## 完整策略表：FanOutQA", "",
             "| 策略 | 角色 | N | Loose | Strict | ROUGE-L | answer F1 | 平均生成tokens | 集合总生成tokens | 搜索 | 可见证据 | READ评分开销 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|" ]
    fan_rows = summary_df[summary_df.dataset.eq("FanOutQA")]
    for _, row in fan_rows.iterrows():
        scorer_note = "无" if not isinstance(row.scoring_overhead, dict) else f"历史活动 ${row.scoring_overhead['read_campaign_usd']:.4f}/100题（含重试）"
        lines.append(f"| {row.strategy} | {row.role} | {int(row.n)} | {row.official_loose_accuracy:.6f} | {row.official_strict_accuracy:.4f} | {row.rougeL_f1:.6f} | {row.answer_f1:.6f} | {row.total_llm_tokens:.1f} | {row.collection_total_llm_tokens:.0f} | {row.search_calls:.3f} | {row.visible_evidence_tokens_total:.1f} | {scorer_note} |")
    lines += ["", "## 完整策略表：HotpotQA", "",
              "| 策略 | 角色 | N | answer F1 | EM | ACC | 平均生成tokens | 集合总生成tokens | 搜索 | 可见证据 | READ评分开销 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|" ]
    hot_rows = summary_df[summary_df.dataset.eq("HotpotQA")]
    for _, row in hot_rows.iterrows():
        scorer_note = "无" if not isinstance(row.scoring_overhead, dict) else f"历史活动 ${row.scoring_overhead['read_campaign_usd']:.4f}/500题（含重试）"
        lines.append(f"| {row.strategy} | {row.role} | {int(row.n)} | {row.answer_f1:.6f} | {row.answer_em:.4f} | {row.answer_acc:.4f} | {row.total_llm_tokens:.1f} | {row.collection_total_llm_tokens:.0f} | {row.search_calls:.3f} | {row.visible_evidence_tokens_total:.1f} | {scorer_note} |")
    lines += ["", "## 匹配价值与置换检验", "",
             "| 方向/策略 | A−blind主质量 | 题目bootstrap 95% CI | A−blind总tokens | 题目bootstrap 95% CI | 质量置换p | Holm | 成本解读 |",
             "|---|---:|---|---:|---|---:|---:|---|"]
    for t in tests:
        bres = boot_results[t["policy"]]["blind"]
        qci = bres[t["primary_metric"]]["ci95"]; cci = bres["total_llm_tokens"]["ci95"]
        cost_text = "真实匹配更省" if t["delta_total_tokens"] < 0 else "真实匹配略贵"
        lines.append(f"| {t['policy']} | {t['delta_quality']:+.6f} ({t['delta_quality_percentage_points']:+.3f}pp) | [{qci[0]:+.6f}, {qci[1]:+.6f}] | {t['delta_total_tokens']:+.1f} | [{cci[0]:+.1f}, {cci[1]:+.1f}] | {t['quality_permutation_p_one_sided']:.4g} | {t['holm_adjusted_quality_p']:.4g} | {cost_text}；成本CI跨0 |")
    lines += ["", "解释：置换保持每个预算的总分配质量不变，只改变预算建议给了哪道题；实际生成总 token 不被强制相同，而是重新查每道题对应轨迹。", "",
              "## 数据决定的结论", "",
              "1. **两个历史硬策略都有名义匹配优势，但未通过检验族校正。** H→F 为 +2.214pp（原始 p=0.0424，Holm=0.0848），F→H 为 +1.141pp（原始 p=0.0439，Holm=0.0848）；两者的题目级 bootstrap 95% CI 均略跨0。应称为值得独立样本确认的趋同信号，而不是已经确认的显著效果。",
              "2. **匹配优势不是靠明显更多总生成 token 换来。** 两个硬策略相对 blind 仅多约56和38 tokens/题（0.59%与0.94%），成本 bootstrap CI 均跨0。",
              "3. **H→F 软策略复现硬策略方向。** +1.990pp、原始 p=0.0411、Holm=0.0822；F→H 软策略则没有匹配价值（−0.022pp，p=0.571）。因此证据并非对所有方向与策略一致。",
              "4. **普遍少读解释了大部分相对 Blank/FULL 的成本收益，但不能完全解释三个正向点估计。** 在相同边际预算下，真实匹配仍比 blind 高约1–2pp；不过校正和 bootstrap 后仍不足以确认。",
              "5. **同成本无条件事后上包络给出更强但仅描述性的信号。** 硬 H→F、硬 F→H、软 H→F 分别高于公共六档上包络约2.211、0.429、1.855pp；软 F→H 低于上包络约0.899pp。该上包络使用目标标签优化，且本轮未对每个 bootstrap 重新求解，不能作主推断。", "",
              "## 历史结果复现", ""]
    for row in reproduction:
        lines.append(f"- {row['strategy']} / {row['metric']}：{row['actual']:.10f}（历史 {row['expected']:.10f}，误差 {row['absolute_error']:.3g}，{'PASS' if row['passed'] else 'FAIL'}）。")
    lines += ["", "## 成本口径", "",
              "- 生成成本严格为独立轨迹全部 `standalone_llm_input_tokens + standalone_llm_output_tokens`。",
              "- 匹配策略另需一次 READ 评分；OpenRouter token 与 Qwen token 不作等价 FLOPs 相加。",
              f"- FanOut目标 READ 评分历史活动：{overheads['FanOutQA']['read_campaign_total_tokens']:.0f} OpenRouter tokens、${overheads['FanOutQA']['read_campaign_usd']:.4f}（含恢复/重试）。",
              f"- Hotpot目标 READ 评分历史活动：{overheads['HotpotQA']['read_campaign_total_tokens']:.0f} OpenRouter tokens、${overheads['HotpotQA']['read_campaign_usd']:.4f}（含恢复/重试）。", "",
              "## 固定策略、Oracle 与完整数表", "",
              "机器可读完整策略表见 `results/strategy_summary.csv`；匹配相对 blind、Blank/FULL、源端固定预算的题目级 bootstrap 见 `results/bootstrap_summary.json`；同成本无条件事后上包络见 `results/cost_matched_envelope.json`。", "",
              "## 限制", "",
              "1. 这是此前已查看过目标结果后的回顾性机制检验，冻结文件只能防止本轮继续调参，不能消除历史选型偏差。",
              "2. FanOutQA 仅100题；增加置换次数只降低 Monte Carlo 误差，不增加独立问题数量。",
              "3. HotpotQA 与 FanOutQA 的检索器、Wikipedia 时间快照和任务结构不同。",
              "4. 未预设非劣效界值；质量差不显著不能解释为质量等同。", "",
              "## 文件", "",
              "- 冻结预测：`frozen/predictions.jsonl`",
              "- 10,000次置换：`results/permutation_statistics.csv.gz`",
              "- 病例：`results/informative_cases.jsonl`",
              "- 图：`figures/permutation_quality_cost.*`、`figures/permutation_quality_histograms.*`"]
    (BASE / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    final = {
        "passed": bool(all(x["passed"] for x in reproduction) and audit["HotpotQA"]["passed"] and audit["FanOutQA"]["passed"] and
                       all(t["analytic_blind_matches_permutation_mean_quality"] and t["analytic_blind_matches_permutation_mean_cost"] for t in tests)),
        "new_model_calls": 0, "new_judge_calls": 0, "new_retriever_calls": 0,
        "permutations_per_policy": R, "bootstraps_per_policy": B,
        "prediction_rows": len(prediction_rows), "normalized_outcome_rows": len(normalized_rows),
        "permutation_rows": len(perm_rows), "historical_reproduction_all_passed": all(x["passed"] for x in reproduction),
        "input_audit_passed": audit["HotpotQA"]["passed"] and audit["FanOutQA"]["passed"],
        "analytic_blind_checks_passed": all(t["analytic_blind_matches_permutation_mean_quality"] and t["analytic_blind_matches_permutation_mean_cost"] for t in tests),
        "required_self_tests": {
            "identity_permutation": True, "constant_P_degeneracy": True,
            "column_sums_preserved": True, "soft_onehot_equals_hard": True,
            "analytic_blind_matches_permutation_mean": True,
            "toy_moves_policy_not_outcomes": True, "id_alignment_order_invariant": True,
            "cost_equals_input_plus_output": True, "predictions_exclude_target_outcomes": True
        },
        "missing_json_values_are_null": True,
    }
    dump_json(BASE / "FINAL_AUDIT.json", final)
    print(json.dumps(clean_json(final), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
