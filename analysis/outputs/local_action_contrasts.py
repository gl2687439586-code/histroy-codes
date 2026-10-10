from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\RESEARCH_USER\Documents\Codex\2026-08-11\referenced-chatgpt-conversation-this-is-an-3")
INPUT = ROOT / "outputs/qwen3_only_bamboogle_v6_review/02_src_tree/analysis_primary_natural_roots/sample_sequence_values.csv"
d = pd.read_csv(INPUT)
d = d[d["horizon"].eq(3)].pivot(index="sample_id", columns="path", values=["f1_mean", "total_cost_tokens_mean"])
rng = np.random.default_rng(2026)

states = {
    "after_S": {"Commit": "SC", "Read": "SRC", "Search": "SSC"},
    "after_SR": {"Commit": "SRC", "Read": "SRR", "Search": "SRS"},
    "after_SS": {"Commit": "SSC", "Read": "SSR", "Search": "SSS"},
}

rows = []
for state, actions in states.items():
    for action, path in actions.items():
        q = d["f1_mean"][path].to_numpy(float)
        c = d["total_cost_tokens_mean"][path].to_numpy(float)
        rows.append({"state": state, "action": action, "path": path, "mean_f1": q.mean(), "mean_cost": c.mean()})
    for action in ("Read", "Search"):
        q = (d["f1_mean"][actions[action]] - d["f1_mean"][actions["Commit"]]).to_numpy(float)
        c = (d["total_cost_tokens_mean"][actions[action]] - d["total_cost_tokens_mean"][actions["Commit"]]).to_numpy(float)
        ix = rng.integers(0, len(q), size=(10000, len(q)))
        rows.append({
            "state": state,
            "action": f"{action}-minus-Commit",
            "path": f"{actions[action]}-{actions['Commit']}",
            "mean_f1": q.mean(),
            "f1_ci_low": np.quantile(q[ix].mean(axis=1), .025),
            "f1_ci_high": np.quantile(q[ix].mean(axis=1), .975),
            "mean_cost": c.mean(),
            "cost_ci_low": np.quantile(c[ix].mean(axis=1), .025),
            "cost_ci_high": np.quantile(c[ix].mean(axis=1), .975),
            "quality_win": int((q > 0).sum()), "quality_tie": int((q == 0).sum()), "quality_loss": int((q < 0).sum()),
        })

    # Local action winner distribution, separated into strict winner versus ties.
    quality = pd.DataFrame({a: d["f1_mean"][p] for a, p in actions.items()})
    maxima = quality.max(axis=1)
    masks = quality.eq(maxima, axis=0)
    for action in actions:
        rows.append({
            "state": state,
            "action": f"strict_unique_best_{action}",
            "path": actions[action],
            "count": int((masks[action] & masks.sum(axis=1).eq(1)).sum()),
            "rate": float((masks[action] & masks.sum(axis=1).eq(1)).mean()),
        })
    rows.append({"state": state, "action": "top_quality_tie", "count": int(masks.sum(axis=1).gt(1).sum()), "rate": float(masks.sum(axis=1).gt(1).mean())})

out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs/step_token_analysis/local_action_contrasts.csv", index=False)
print(out.round(4).to_string(index=False))

# Order effect: same action multiset {S,S,R}, different order after the fixed S.
order_q = (d["f1_mean"]["SSR"] - d["f1_mean"]["SRS"]).to_numpy(float)
order_c = (d["total_cost_tokens_mean"]["SSR"] - d["total_cost_tokens_mean"]["SRS"]).to_numpy(float)
ix = rng.integers(0, len(order_q), size=(10000, len(order_q)))

# State-action interaction: the same next action has different value depending on whether
# the previous decision was Read or Search.
search_after_read = d["f1_mean"]["SRS"] - d["f1_mean"]["SRC"]
search_after_search = d["f1_mean"]["SSS"] - d["f1_mean"]["SSC"]
read_after_search = d["f1_mean"]["SSR"] - d["f1_mean"]["SSC"]
read_after_read = d["f1_mean"]["SRR"] - d["f1_mean"]["SRC"]
interactions = {
    "Search value: after Read minus after Search": (search_after_read - search_after_search).to_numpy(float),
    "Read value: after Search minus after Read": (read_after_search - read_after_read).to_numpy(float),
}
extra = [{
    "contrast": "SSR minus SRS (same counts, different order)",
    "mean_quality_difference": order_q.mean(),
    "quality_ci_low": np.quantile(order_q[ix].mean(axis=1), .025),
    "quality_ci_high": np.quantile(order_q[ix].mean(axis=1), .975),
    "mean_cost_difference": order_c.mean(),
    "cost_ci_low": np.quantile(order_c[ix].mean(axis=1), .025),
    "cost_ci_high": np.quantile(order_c[ix].mean(axis=1), .975),
}]
for name, values in interactions.items():
    ix = rng.integers(0, len(values), size=(10000, len(values)))
    extra.append({
        "contrast": name,
        "mean_quality_difference": values.mean(),
        "quality_ci_low": np.quantile(values[ix].mean(axis=1), .025),
        "quality_ci_high": np.quantile(values[ix].mean(axis=1), .975),
    })
extra = pd.DataFrame(extra)
extra.to_csv(ROOT / "outputs/step_token_analysis/action_order_and_interactions.csv", index=False)
print("\n", extra.round(4).to_string(index=False))
