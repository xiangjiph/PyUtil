from __future__ import annotations

import json
from pathlib import Path
import time

from pyutil.sys import SlurmJobClient


def _wait_until_finished(client: SlurmJobClient, task_dir: Path, timeout_sec: int = 240) -> dict:
    start = time.time()
    while time.time() - start < timeout_sec:
        status = client.collect_task_status(str(task_dir))
        summary = status["summary"]
        if summary["failed"]:
            return status
        if summary["complete"]:
            return status
        time.sleep(5)
    raise TimeoutError("Timed out waiting for task completion")


def run_integration_test() -> None:
    root = Path("/scratch/xj0996/slurm_tasks")
    client = SlurmJobClient(root_folder=root, default_conda_env="common")

    task_name = "slurm_client_dict_args"
    task_stamp = client.timestamp()
    task_dir = client.create_task_folder(task_name, task_stamp=task_stamp)

    rows = [
        {"cell": "A", "weight": 1},
        {"cell": "B", "weight": 2},
        {"cell": "C", "weight": 3},
        {"cell": "D", "weight": 4},
    ]
    args_file = task_dir / "shared" / "args.txt"
    args_file.write_text("\n".join(json.dumps(item, sort_keys=True) for item in rows) + "\n")

    worker_script = task_dir / "shared" / "dict_args_worker.py"
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
    print(
        f"NODE={os.environ.get('HOSTNAME', 'unknown')} "
        f"CELL={payload['cell']} WEIGHT={payload['weight']}"
    )


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
    status = _wait_until_finished(client, task_dir)
    if status["summary"]["failed"]:
        raise RuntimeError(f"Unexpected failures: {status['summary']['failed']}")

    log_files = sorted((task_dir / "logs").glob("*.out"))
    if len(log_files) < len(rows):
        raise AssertionError("Expected one or more .out logs per array task")

    merged = "\n".join(path.read_text() for path in log_files)
    for item in rows:
        token = f"CELL={item['cell']} WEIGHT={item['weight']}"
        if token not in merged:
            raise AssertionError(f"Missing expected output token: {token}")

    removed = client.cleanup_error_logs_if_no_error(str(task_dir))
    if not removed:
        raise AssertionError("Expected clean-run .err files to be removed")
    if list((task_dir / "logs").glob("*.err")):
        raise AssertionError("Expected no .err files after cleanup")

    print("Dict args array integration test passed.")
    print(f"Task directory: {task_dir}")


if __name__ == "__main__":
    run_integration_test()
