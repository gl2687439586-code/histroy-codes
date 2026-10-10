import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\RESEARCH_USER\Documents\Codex\2026-08-11\referenced-chatgpt-conversation-this-is-an-3")
INPUT = ROOT / "outputs/qwen3_only_bamboogle_v6_review/02_src_tree/analysis_primary_natural_roots/sample_sequence_values.csv"
OUTPUT = ROOT / "outputs/pareto_frontier_h3_stats.json"

d = pd.read_csv(INPUT)
d = d[d["horizon"].eq(3)].copy()
rng = np.random.default_rng(2026)
rows = []

for path, g in d.groupby("path"):
    cost = g["total_cost_tokens_mean"].to_numpy(float)
    quality = g["f1_mean"].to_numpy(float)
    ix = rng.integers(0, len(g), size=(10000, len(g)))
    rows.append(
        {
            "path": path,
            "n": len(g),
            "cost": round(float(cost.mean()), 3),
            "cost_ci95": np.quantile(cost[ix].mean(axis=1), [0.025, 0.975]).round(3).tolist(),
            "f1": round(float(quality.mean()), 6),
            "f1_ci95": np.quantile(quality[ix].mean(axis=1), [0.025, 0.975]).round(6).tolist(),
        }
    )

pivot = d.pivot(index="sample_id", columns="path", values=["f1_mean", "total_cost_tokens_mean"])
quality_diff = (pivot["f1_mean"]["SSC"] - pivot["f1_mean"]["SSS"]).to_numpy(float)
cost_diff = (pivot["total_cost_tokens_mean"]["SSC"] - pivot["total_cost_tokens_mean"]["SSS"]).to_numpy(float)
ix = rng.integers(0, len(quality_diff), size=(10000, len(quality_diff)))

result = {
    "points": sorted(rows, key=lambda row: row["cost"]),
    "ssc_minus_sss": {
        "f1": round(float(quality_diff.mean()), 6),
        "f1_ci95": np.quantile(quality_diff[ix].mean(axis=1), [0.025, 0.975]).round(6).tolist(),
        "cost": round(float(cost_diff.mean()), 3),
        "cost_ci95": np.quantile(cost_diff[ix].mean(axis=1), [0.025, 0.975]).round(3).tolist(),
    },
}
OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
