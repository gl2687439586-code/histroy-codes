#!/usr/bin/env python3
"""Filter terminal trajectories by audited root origin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root_origin", default="qwen_natural_root")
    args = parser.parse_args()
    source = Path(args.input)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("root_origin") == args.root_origin:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(json.dumps({"input": str(source), "output": str(target), "root_origin": args.root_origin, "records": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

