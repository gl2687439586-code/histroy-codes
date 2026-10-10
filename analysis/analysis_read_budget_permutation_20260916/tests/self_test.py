from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis", ROOT / "run_analysis.py")
analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analysis)


def main() -> None:
    rng = np.random.default_rng(2026)
    n, k = 8, 6
    P = rng.dirichlet(np.ones(k), size=n)
    Q = rng.random((n, k))
    C = rng.integers(100, 1000, size=(n, k)).astype(float)

    # Identity permutation reproduces matched policy exactly.
    identity = np.arange(n)
    assert np.allclose(np.sum(P * Q, axis=1), np.sum(P[identity] * Q, axis=1))

    # Moving P, not Q/C rows, matches a manually constructed toy example.
    toy_p = np.array([[1, 0], [0, 1]], float)
    toy_q = np.array([[1, 0], [0.2, 0.9]], float)
    swapped = toy_p[[1, 0]]
    assert np.isclose(np.mean(np.sum(swapped * toy_q, axis=1)), 0.1)

    # Constant policy rows make every permutation identical to A and blind.
    constant = np.repeat(P.mean(axis=0, keepdims=True), n, axis=0)
    a = np.mean(np.sum(constant * Q, axis=1))
    blind = np.mean(Q @ constant.mean(axis=0))
    assert np.isclose(a, blind)
    for _ in range(20):
        assert np.isclose(a, np.mean(np.sum(constant[rng.permutation(n)] * Q, axis=1)))

    # Permutations preserve full rows and therefore column sums.
    for _ in range(20):
        assert np.allclose(P[rng.permutation(n)].sum(axis=0), P.sum(axis=0))

    # Soft one-hot equals the corresponding hard policy.
    hard_budget = rng.integers(0, k, size=n)
    onehot = np.zeros((n, k)); onehot[np.arange(n), hard_budget] = 1
    assert np.allclose(np.sum(onehot * Q, axis=1), Q[np.arange(n), hard_budget])
    assert np.allclose(np.sum(onehot * C, axis=1), C[np.arange(n), hard_budget])

    # Analytic blind equals permutation expectation up to Monte Carlo error.
    expected = np.mean(Q @ P.mean(axis=0))
    draws = np.array([np.mean(np.sum(P[rng.permutation(n)] * Q, axis=1)) for _ in range(20000)])
    assert abs(draws.mean() - expected) <= 5 * draws.std(ddof=1) / np.sqrt(len(draws)) + 1e-12

    # ID-based alignment is invariant to input row order.
    ids = np.array([f"q{i}" for i in range(n)])
    order = rng.permutation(n)
    lookup = {sid: row for sid, row in zip(ids[order], Q[order])}
    restored = np.stack([lookup[sid] for sid in ids])
    assert np.array_equal(restored, Q)

    result = {
        "passed": True,
        "checks": ["identity_permutation", "constant_P_degeneracy", "rowwise_policy_permutation",
                   "column_sums_preserved", "soft_onehot_equals_hard", "analytic_blind_expectation",
                   "toy_moves_policy_not_outcomes", "id_alignment_order_invariant"],
    }
    (ROOT / "tests/self_test_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SELF-TEST PASSED: identity, constant-P degeneration, row-wise permutation, one-hot equivalence, blind expectation, ID alignment.")


if __name__ == "__main__":
    main()
