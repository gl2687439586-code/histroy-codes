"""Export finished repeat results, raw trajectories and retrieval evidence."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import tarfile


def export_review(batch):
    batch = Path(batch).resolve()
    archive = batch.parent / (batch.name + "_review_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".tar.gz")
    files = [batch / name for name in ("BATCH_CONFIG.json", "BATCH_STATUS.json", "last_service_check.json")]
    files += list((batch / "analysis_repeats").rglob("*"))
    files += list((batch / "tmux_logs").glob("*.log"))
    for repetition in range(1, 4):
        run = batch / f"repeat_{repetition}"
        files += [run / name for name in ("REPEAT_CONFIG.json", "PIPELINE_STATUS.json", "last_service_check.json")]
        for directory in ("dataset", "analysis", "tmux_logs"):
            files += list((run / directory).rglob("*"))
        for stage in ("roots", "trajectories"):
            files += list((run / stage).glob("*.gz*"))
            for shard in (run / stage / "shards").glob("shard_*"):
                files += [shard / name for name in ("manifest.json", "run_status.json", "failures.jsonl.gz")]
    with tarfile.open(archive, "x:gz") as handle:
        for path in sorted(set(files)):
            if path.is_file():
                handle.add(path, arcname=str(path.relative_to(batch)), recursive=False)
    return str(archive)
