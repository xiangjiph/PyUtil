from __future__ import annotations

from pathlib import Path
import time
from typing import Dict

from pyutil.sys import SlurmJobClient


def wait_for_completion(
    client: SlurmJobClient,
    task_dir: Path,
    timeout_sec: int = 240,
    require_failed: bool = False,
) -> Dict[str, object]:
    """Poll task status until completion or failure criteria are met."""
    start = time.time()
    while time.time() - start < timeout_sec:
        status = client.collect_task_status(str(task_dir))
        summary = status["summary"]
        if require_failed:
            if summary["failed"] and summary["complete"]:
                return status
        else:
            if summary["failed"] or summary["complete"]:
                return status
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for SLURM task in {task_dir}")
