#!/usr/bin/env python3
"""Low-overhead RAM, swap, disk, GPU, and experiment-progress reminder."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GIB = 1024 ** 3


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path):
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # A writer may currently be appending the final line.
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    sample_id = str(record.get("sample_id", record.get("id", "")))
    rollout_id = int(record.get("rollout_id", 0) or 0)
    return sample_id, rollout_id


def unique_records(path: Path) -> int:
    return len({record_key(row) for row in read_jsonl(path) or () if record_key(row)[0]})


def complete_trees(path: Path) -> int:
    counts: dict[tuple[str, int], int] = defaultdict(int)
    expected: dict[tuple[str, int], int] = {}
    for row in read_jsonl(path) or ():
        key = record_key(row)
        if not key[0]:
            continue
        counts[key] += 1
        raw_expected = row.get("expected_terminal_paths")
        if raw_expected is None:
            horizon = int(row.get("horizon", 3) or 3)
            raw_expected = (2 ** horizon) - 1
        expected[key] = int(raw_expected)
    return sum(counts[key] >= expected.get(key, 1) for key in counts)


def progress_snapshot(output_base: Path) -> dict[str, int]:
    values = {
        "blank_questions": unique_records(output_base / "01_blank" / "trajectories.jsonl"),
        "root_checkpoints": unique_records(output_base / "roots" / "qwen3_first_search_checkpoints.jsonl"),
        "complete_action_trees": complete_trees(output_base / "02_src_tree" / "terminal_trajectories.jsonl"),
        "neutral_questions": unique_records(output_base / "03_neutral" / "trajectories.jsonl"),
    }
    values["completed_question_stage_units"] = sum(values.values())
    return values


def mem_snapshot() -> dict[str, float]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
    swap_used = values.get("SwapTotal", 0) - values.get("SwapFree", 0)
    return {
        "total_gib": round(values.get("MemTotal", 0) / GIB, 2),
        "available_gib": round(values.get("MemAvailable", 0) / GIB, 2),
        "swap_used_gib": round(swap_used / GIB, 2),
    }


def disk_snapshot(path: Path) -> dict[str, float | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gib": round(usage.total / GIB, 2),
        "free_gib": round(usage.free / GIB, 2),
        "used_percent": round(100.0 * usage.used / usage.total, 2),
    }


def gpu_snapshot() -> list[dict[str, float | int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, float | int]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, used, total, utilization = map(float, parts)
        rows.append(
            {
                "index": int(index),
                "memory_used_mib": round(used, 1),
                "memory_total_mib": round(total, 1),
                "memory_used_percent": round(100.0 * used / total, 2) if total else 0.0,
                "utilization_percent": round(utilization, 1),
            }
        )
    return rows


def parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def make_snapshot(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    ram = mem_snapshot()
    disk = disk_snapshot(args.work_root)
    gpus = gpu_snapshot()
    warnings: list[str] = []
    if ram["available_gib"] < args.min_available_ram_gib:
        warnings.append(
            f"available RAM {ram['available_gib']}GiB < {args.min_available_ram_gib}GiB"
        )
    if ram["swap_used_gib"] > args.max_swap_used_gib:
        warnings.append(f"swap used {ram['swap_used_gib']}GiB > {args.max_swap_used_gib}GiB")
    if disk["free_gib"] < args.min_free_disk_gib:
        warnings.append(f"free disk {disk['free_gib']}GiB < {args.min_free_disk_gib}GiB")
    for gpu in gpus:
        if gpu["memory_used_percent"] > args.max_gpu_memory_percent:
            warnings.append(
                f"GPU {gpu['index']} memory {gpu['memory_used_percent']}% > "
                f"{args.max_gpu_memory_percent}%"
            )
    return {
        "created_at": utc_now(),
        "reason": reason,
        "progress": progress_snapshot(args.output_base),
        "ram": ram,
        "disk": disk,
        "gpus": gpus,
        "warnings": warnings,
        "status": "warning" if warnings else "ok",
    }


def emit(snapshot: dict[str, Any], log_path: Path, warning_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
    progress = snapshot["progress"]
    ram = snapshot["ram"]
    disk = snapshot["disk"]
    gpu_text = ", ".join(
        f"GPU{gpu['index']}={gpu['memory_used_mib']:.0f}/{gpu['memory_total_mib']:.0f}MiB"
        for gpu in snapshot["gpus"]
    ) or "GPU data unavailable"
    prefix = "RESOURCE WARNING" if snapshot["warnings"] else "RESOURCE REMINDER"
    print(
        f"\n[{snapshot['created_at']}] === {prefix} ({snapshot['reason']}) ===\n"
        f"progress: blank={progress['blank_questions']} roots={progress['root_checkpoints']} "
        f"trees={progress['complete_action_trees']} neutral={progress['neutral_questions']}\n"
        f"RAM available={ram['available_gib']:.2f}/{ram['total_gib']:.2f}GiB; "
        f"swap used={ram['swap_used_gib']:.2f}GiB\n"
        f"disk free={disk['free_gib']:.2f}/{disk['total_gib']:.2f}GiB at {disk['path']}\n"
        f"{gpu_text}",
        flush=True,
    )
    if snapshot["warnings"]:
        warning_path.write_text(encoded + "\n", encoding="utf-8")
        print("\aACTION REQUIRED: " + "; ".join(snapshot["warnings"]), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--min-available-ram-gib", type=float, default=64)
    parser.add_argument("--max-swap-used-gib", type=float, default=16)
    parser.add_argument("--min-free-disk-gib", type=float, default=150)
    parser.add_argument("--max-gpu-memory-percent", type=float, default=95)
    parser.add_argument("--stop-on-critical", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)
    log_path = args.output_base / "resource_monitor.jsonl"
    warning_path = args.output_base / "RESOURCE_WARNING.flag"
    last_milestone = -1
    last_emit_time = 0.0
    first = True
    while parent_alive(args.parent_pid):
        progress = progress_snapshot(args.output_base)
        units = progress["completed_question_stage_units"]
        milestone = units // max(args.progress_every, 1)
        now = time.monotonic()
        interval_due = (now - last_emit_time) >= max(args.interval_seconds, 30)
        milestone_due = not first and milestone > last_milestone
        if first or interval_due or milestone_due:
            reason = "startup" if first else "periodic"
            if milestone_due:
                reason = f"progress milestone: {units} completed question-stage units"
            snapshot = make_snapshot(args, reason)
            emit(snapshot, log_path, warning_path)
            last_milestone = milestone
            last_emit_time = now
            first = False
            if snapshot["warnings"] and args.stop_on_critical and args.parent_pid > 0:
                print("RESOURCE_STOP_ON_CRITICAL=1: terminating the experiment wrapper.", flush=True)
                os.kill(args.parent_pid, signal.SIGTERM)
                return 3
            if args.once:
                return 0 if not snapshot["warnings"] else 1
        # Poll progress cheaply so a 10-unit milestone is reported within 30s,
        # while full resource snapshots still occur at the configured interval.
        time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
