from __future__ import annotations

from pathlib import Path

from pyutil.sys import SlurmJobClient
from slurm_test_utils import wait_for_completion


def run_integration_test() -> None:
    root = Path("/scratch/xj0996/slurm_tasks")
    client = SlurmJobClient(root_folder=root, default_conda_env="common")

    task_name = "slurm_client_smoke"
    task_stamp = client.timestamp()
    task_dir = client.create_task_folder(task_name, task_stamp=task_stamp)

    arg_file = task_dir / "shared" / "args.txt"
    arg_file.write_text("alpha 1\nbeta 2\n")

    worker_script = task_dir / "shared" / "echo_args_worker.py"
    worker_script.write_text(
        """
from pathlib import Path
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    lines = Path(args.cfg).read_text().splitlines()
    selected = lines[args.start:args.end]
    for value in selected:
        print(f"NODE={os.environ.get('HOSTNAME', 'unknown')} ARG={value}")


if __name__ == "__main__":
    main()
""".strip()
        + "\n"
    )

    submit_payload = client.submit_index_range_test_unit(
        task_name=task_name,
        worker_script=str(worker_script),
        config_path=str(arg_file),
        unit_start=0,
        unit_end=2,
        task_stamp=task_stamp,
        submit=True,
        nodes=1,
        cpus_per_task=1,
        mem_per_cpu_gb=1,
        wall_time="00:01:00",
        conda_env="common",
    )

    print(f"Submitted job: {submit_payload['job_id']}")
    status = wait_for_completion(client, task_dir)

    failed = status["summary"]["failed"]
    if failed:
        raise RuntimeError(f"Job failed: {failed}")

    removed_err = client.cleanup_error_logs_if_no_error(str(task_dir))
    if not removed_err:
        raise AssertionError("Expected empty .err files to be removed for a clean run")
    if list((task_dir / "logs").glob("*.err")):
        raise AssertionError("Expected no .err files after cleanup")

    log_files = sorted((task_dir / "logs").glob("*.out"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        raise FileNotFoundError(f"No SLURM .out logs found in {task_dir / 'logs'}")

    latest_log = log_files[-1]
    text = latest_log.read_text()

    expected_tokens = ["ARG=alpha 1", "ARG=beta 2", "NODE="]
    missing = [token for token in expected_tokens if token not in text]
    if missing:
        raise AssertionError(f"Missing expected log content: {missing}\nLog:\n{text}")

    print("Integration test passed.")
    print(f"Task directory: {task_dir}")
    print(f"Verified log file: {latest_log}")


if __name__ == "__main__":
    run_integration_test()
