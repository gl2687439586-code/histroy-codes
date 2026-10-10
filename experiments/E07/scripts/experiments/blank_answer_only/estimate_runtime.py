#!/usr/bin/env python3
"""Estimate initial and live wall-clock time for the four-shard experiment."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def quantile(values: List[float], probability: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def answer_latency(row: Dict[str, Any]) -> float:
    attempts = (row.get("generation_attempts") or {}).get("answer") or []
    if attempts:
        return sum(float(item.get("latency_seconds", 0)) for item in attempts)
    return float(row.get("latency_seconds", 0))


def initial(args: argparse.Namespace) -> Dict[str, Any]:
    pilot = read_jsonl(Path(args.pilot_records)) if args.pilot_records else []
    values = [answer_latency(row) for row in pilot if answer_latency(row) > 0]
    assumed = sum(values) / len(values) if values else args.fallback_seconds_per_record
    base_wall = assumed * args.total_records / args.workers
    return {
        "kind": "initial_runtime_estimate",
        "pilot_records": len(values),
        "pilot_mean_answer_seconds": assumed,
        "pilot_p50_answer_seconds": quantile(values, 0.5) if values else assumed,
        "pilot_p90_answer_seconds": quantile(values, 0.9) if values else assumed,
        "total_records": args.total_records,
        "workers": args.workers,
        "expected_model_calls": args.total_records,
        "base_estimated_wall_hours": base_wall / 3600,
        "safety_multiplier": args.safety_multiplier,
        "recommended_planning_wall_hours": (
            base_wall * args.safety_multiplier / 3600
        ),
        "assumptions": [
            "one model request per record",
            "four independent one-GPU workers",
            "pilot latency includes native Qwen3 thinking",
            "live ETA supersedes this estimate after records complete",
        ],
    }


def progress(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir)
    records: List[Dict[str, Any]] = []
    failures = 0
    for shard in range(args.workers):
        base = run_dir / "shards" / f"shard_{shard}"
        for path in (base / "record_parts").glob("*.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        failures += sum(1 for _ in (base / "failure_parts").glob("*.json"))
    now = time.time()
    started = float(args.started_epoch)
    elapsed = max(0.0, now - started)
    completed = len(records)
    wall_rate = completed / elapsed if completed and elapsed else 0.0
    remaining = max(0, args.total_records - completed)
    live_eta = remaining / wall_rate if wall_rate else None
    latencies = [float(row.get("latency_seconds", 0)) for row in records]
    result = {
        "kind": "live_runtime_estimate",
        "timestamp_epoch": now,
        "elapsed_hours": elapsed / 3600,
        "records_completed": completed,
        "records_remaining": remaining,
        "completion_fraction": completed / args.total_records,
        "transport_failures_present": failures,
        "aggregate_throughput_records_per_hour": wall_rate * 3600,
        "live_eta_hours": live_eta / 3600 if live_eta is not None else None,
        "live_estimated_total_wall_hours": (
            (elapsed + live_eta) / 3600 if live_eta is not None else None
        ),
        "mean_completed_record_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "invalid_format_records": sum(
            row.get("answer_status") != "valid" for row in records
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("initial", "progress"))
    parser.add_argument("--pilot_records", default="")
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--total_records", type=int, default=1875)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--safety_multiplier", type=float, default=1.25)
    parser.add_argument("--fallback_seconds_per_record", type=float, default=35.0)
    parser.add_argument("--started_epoch", type=float, default=0.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = initial(args) if args.mode == "initial" else progress(args)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
