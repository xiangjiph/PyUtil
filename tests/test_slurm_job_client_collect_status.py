from __future__ import annotations

import json
from pathlib import Path
import time

from pyutil.sys import SlurmJobClient


def _wait_for_mixed_outcome(client: SlurmJobClient, task_dir: Path, timeout_sec: int = 300) -> dict:
    start = time.time()
    while time.time() - start < timeout_sec:
        status = client.collect_task_status(str(task_dir), include_log_errors=True)
        summary = status["summary"]
        if summary["failed"] and summary["complete"]:
            return status
        time.sleep(5)
    raise TimeoutError("Timed out waiting for mixed complete/failed status")


def run_integration_test() -> None:
    root = Path("/scratch/xj0996/slurm_tasks")
    client = SlurmJobClient(root_folder=root, default_conda_env="common")

    task_name = "slurm_client_collect_status"
    task_stamp = client.timestamp()
    task_dir = client.create_task_folder(task_name, task_stamp=task_stamp)

    rows = [
        {"id": 1, "fail": False},
        {"id": 2, "fail": True},
        {"id": 3, "fail": False},
    ]
    args_file = task_dir / "shared" / "args.txt"
    args_file.write_text("\n".join(json.dumps(item, sort_keys=True) for item in rows) + "\n")

    worker_script = task_dir / "shared" / "status_worker.py"
    worker_script.write_text(
        """
import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg-json", type=str, required=True)
    args = parser.parse_args()

    payload = json.loads(args.arg_json)
    print(f"NODE={os.environ.get('HOSTNAME', 'unknown')} ID={payload['id']}")
    if payload["fail"]:
        raise RuntimeError(f"intentional failure for {payload['id']}")


if __name__ == "__main__":
    main()
""".strip()
        + "\n"
    )

    submit_payload = client.submit_args_file_python_array(
        task_name=task_name,
        worker_script=str(worker_script),
        args_file=str(args_file),
        total_items=len(rows),
        task_stamp=task_stamp,
        submit=True,
        nodes=2,
        cpus_per_task=1,
        mem_per_cpu_gb=1,
        wall_time="00:02:00",
        conda_env="common",
    )

    print(f"Submitted job: {submit_payload['job_id']}")
    status = _wait_for_mixed_outcome(client, task_dir)

    failed_jobs = client.find_failed_jobs(str(task_dir))
    if not failed_jobs:
        raise AssertionError("Expected failed jobs but none were reported")

    complete_jobs = status["summary"]["complete"]
    if not complete_jobs:
        raise AssertionError("Expected completed jobs but none were reported")

    err_files = sorted((task_dir / "logs").glob("*.err"))
    if not err_files:
        raise AssertionError("Expected .err files for failed tasks")

    err_text = "\n".join(path.read_text() for path in err_files)
    if "intentional failure" not in err_text:
        raise AssertionError("Did not find expected failure signature in .err logs")

    removed = client.cleanup_error_logs_if_no_error(str(task_dir))
    if removed:
        raise AssertionError("Error logs should not be removed when failures exist")

    print("collect_task_status integration test passed.")
    print(f"Task directory: {task_dir}")


if __name__ == "__main__":
    run_integration_test()
