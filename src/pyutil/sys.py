from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import subprocess
import textwrap
from typing import Dict, List, Optional


task_root_folder = Path("/scratch/xj0996/slurm_tasks")


@dataclass
class SlurmComputeConfig:
    """Cluster resource and SBATCH configuration."""

    job_name: str
    log_dir: Path
    nodes: int = 1
    cpus_per_task: int = 1
    mem_per_cpu_gb: float = 1.0
    wall_time: str = "00:05:00"
    partition: Optional[str] = None
    qos: Optional[str] = None
    account: Optional[str] = None
    array_spec: Optional[str] = None
    mail_type: Optional[str] = None
    mail_user: Optional[str] = None
    extra_sbatch_lines: Optional[List[str]] = None
    modules: Optional[List[str]] = None
    conda_env: Optional[str] = None


@dataclass
class SlurmJobClient:
    """Utility class for creating, submitting, and tracking SLURM jobs."""

    root_folder: Path = task_root_folder
    default_conda_env: str = "common"

    def __post_init__(self) -> None:
        self.root_folder = Path(self.root_folder)
        self.root_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _run_command(command: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(command, capture_output=True, text=True, check=False)

    @staticmethod
    def _quote_args(args: Optional[List[str]]) -> str:
        if not args:
            return ""
        return " ".join(shlex.quote(str(a)) for a in args)

    @staticmethod
    def _slurm_state_bucket(state: str) -> str:
        normalized = state.upper().split("+")[0]
        if normalized in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "RESIZING", "SUSPENDED"}:
            return "running"
        if normalized in {"COMPLETED"}:
            return "complete"
        if normalized in {
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
            "REVOKED",
        }:
            return "failed"
        return "unknown"

    @staticmethod
    def _parse_array_count(array_spec: Optional[str]) -> Optional[int]:
        if not array_spec:
            return None
        if "," in array_spec:
            return None
        if "-" not in array_spec:
            try:
                return 1
            except ValueError:
                return None

        range_part = array_spec.split("%", 1)[0]
        if "-" not in range_part:
            return None
        start_txt, end_txt = range_part.split("-", 1)
        try:
            start = int(start_txt)
            end = int(end_txt)
        except ValueError:
            return None
        if end < start:
            return None
        return end - start + 1

    @staticmethod
    def _err_file_has_error(path: Path) -> bool:
        text = path.read_text().strip()
        if not text:
            return False
        lowered = text.lower()
        markers = ["traceback", "error", "exception", "failed", "segmentation fault"]
        return any(marker in lowered for marker in markers)

    def create_task_folder(self, task_name: str, task_stamp: Optional[str] = None) -> Path:
        """Create task directory under root_folder as task_name_timestamp."""
        stamp = task_stamp or self.timestamp()
        task_dir = self.root_folder / f"{task_name}_{stamp}"
        (task_dir / "logs").mkdir(parents=True, exist_ok=True)
        (task_dir / "shared").mkdir(parents=True, exist_ok=True)
        (task_dir / "scripts").mkdir(parents=True, exist_ok=True)
        return task_dir

    def configure_compute(
        self,
        *,
        job_name: str,
        log_dir: Path,
        nodes: int = 1,
        cpus_per_task: int = 1,
        mem_per_cpu_gb: float = 1.0,
        wall_time: str = "00:05:00",
        partition: Optional[str] = None,
        qos: Optional[str] = None,
        account: Optional[str] = None,
        array_spec: Optional[str] = None,
        mail_type: Optional[str] = None,
        mail_user: Optional[str] = None,
        extra_sbatch_lines: Optional[List[str]] = None,
        modules: Optional[List[str]] = None,
        conda_env: Optional[str] = None,
    ) -> SlurmComputeConfig:
        """Create one reusable compute configuration for a submission."""
        return SlurmComputeConfig(
            job_name=job_name,
            log_dir=Path(log_dir),
            nodes=nodes,
            cpus_per_task=cpus_per_task,
            mem_per_cpu_gb=mem_per_cpu_gb,
            wall_time=wall_time,
            partition=partition,
            qos=qos,
            account=account,
            array_spec=array_spec,
            mail_type=mail_type,
            mail_user=mail_user,
            extra_sbatch_lines=extra_sbatch_lines,
            modules=modules,
            conda_env=conda_env,
        )

    def build_generic_sbatch_header(self, config: SlurmComputeConfig) -> str:
        """Build generic SBATCH settings from a compute config."""
        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={config.job_name}",
            f"#SBATCH --nodes={config.nodes}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --mem-per-cpu={int(config.mem_per_cpu_gb * 1024)}M",
            f"#SBATCH --time={config.wall_time}",
            f"#SBATCH --output={config.log_dir}/%A_%a.out",
            f"#SBATCH --error={config.log_dir}/%A_%a.err",
        ]
        if config.array_spec:
            sbatch_lines.append(f"#SBATCH --array={config.array_spec}")
        if config.partition:
            sbatch_lines.append(f"#SBATCH --partition={config.partition}")
        if config.qos:
            sbatch_lines.append(f"#SBATCH --qos={config.qos}")
        if config.account:
            sbatch_lines.append(f"#SBATCH --account={config.account}")
        if config.mail_type:
            sbatch_lines.append(f"#SBATCH --mail-type={config.mail_type}")
        if config.mail_user:
            sbatch_lines.append(f"#SBATCH --mail-user={config.mail_user}")
        if config.extra_sbatch_lines:
            sbatch_lines.extend(config.extra_sbatch_lines)

        bootstrap_lines: List[str] = []
        if config.modules:
            bootstrap_lines.extend(f"module load {m}" for m in config.modules)
        selected_env = config.conda_env or self.default_conda_env
        bootstrap_lines.extend(["source ~/.bashrc", f"conda activate {selected_env}"])

        return "\n".join(sbatch_lines + [""] + bootstrap_lines).strip() + "\n"

    @staticmethod
    def compose_sbatch_script(generic_header: str, task_body: str) -> str:
        """Concatenate generic settings and task-specific execution body."""
        return generic_header.rstrip() + "\n\n" + task_body.strip() + "\n"

    @staticmethod
    def build_python_task_body(python_script: str, script_args: Optional[List[str]] = None) -> str:
        """Build task body for one Python script with optional arguments."""
        arg_text = SlurmJobClient._quote_args(script_args)
        return f"python {shlex.quote(str(python_script))}{(' ' + arg_text) if arg_text else ''}"

    @staticmethod
    def build_index_range_task_body(
        worker_script: str,
        config_path: str,
        total_items: int,
        chunk_size: int,
    ) -> str:
        """Build task body for --start --end --cfg style workers."""
        return textwrap.dedent(
            f"""
            CHUNK={chunk_size}
            TOTAL_ITEMS={total_items}
            START=$((SLURM_ARRAY_TASK_ID * CHUNK))
            END=$((START + CHUNK))
            if [ "$END" -gt "$TOTAL_ITEMS" ]; then
              END=$TOTAL_ITEMS
            fi

            python {shlex.quote(str(worker_script))} --start ${{START}} --end ${{END}} --cfg {shlex.quote(str(config_path))}
            """
        ).strip()

    @staticmethod
    def build_single_index_range_body(worker_script: str, config_path: str, start: int, end: int) -> str:
        """Build task body for one index-range unit test job."""
        return (
            f"python {shlex.quote(str(worker_script))} --start {start} "
            f"--end {end} --cfg {shlex.quote(str(config_path))}"
        )

    @staticmethod
    def build_args_file_line_task_body(
        worker_script: str,
        args_file: str,
        *,
        line_argument_name: str = "--arg-json",
        worker_extra_args: Optional[List[str]] = None,
    ) -> str:
        """Build array-task body that maps one args file line to one array index."""
        quoted_extra = SlurmJobClient._quote_args(worker_extra_args)
        return textwrap.dedent(
            f"""
            ARGS_FILE={shlex.quote(str(args_file))}
            LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ARGS_FILE")
            if [ -z "$LINE" ]; then
              echo "No argument line for task index $SLURM_ARRAY_TASK_ID" >&2
              exit 2
            fi

            python {shlex.quote(str(worker_script))} {line_argument_name} "$LINE"{(' ' + quoted_extra) if quoted_extra else ''}
            """
        ).strip()

    def write_sbatch_script(self, task_dir: Path, script_name: str, script_text: str) -> Path:
        script_path = task_dir / "scripts" / script_name
        script_path.write_text(script_text)
        return script_path

    def submit_sbatch(self, sbatch_path: Path, submit: bool = True) -> Dict[str, Optional[str]]:
        """Submit sbatch script and return submission metadata."""
        if not submit:
            return {"sbatch_path": str(sbatch_path), "job_id": None, "stdout": "dry-run"}

        result = self._run_command(["sbatch", str(sbatch_path)])
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

        tokens = result.stdout.strip().split()
        job_id = tokens[-1] if tokens else ""
        return {"sbatch_path": str(sbatch_path), "job_id": job_id, "stdout": result.stdout.strip()}

    @staticmethod
    def _to_json_serializable(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): SlurmJobClient._to_json_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SlurmJobClient._to_json_serializable(v) for v in value]
        if isinstance(value, tuple):
            return [SlurmJobClient._to_json_serializable(v) for v in value]
        return value

    @staticmethod
    def _persist_submission(task_dir: Path, payload: Dict[str, object]) -> None:
        metadata_path = task_dir / "submission.json"
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text())
        else:
            existing = {"jobs": []}
        existing.setdefault("jobs", []).append(SlurmJobClient._to_json_serializable(payload))
        metadata_path.write_text(json.dumps(existing, indent=2, sort_keys=True))

    def submit_task_body(
        self,
        *,
        task_name: str,
        script_name: str,
        task_body: str,
        task_stamp: Optional[str] = None,
        submit: bool = True,
        compute: Optional[SlurmComputeConfig] = None,
        compute_overrides: Optional[Dict[str, object]] = None,
        mode: str = "generic",
        metadata_extra: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Submit any task-specific body using one centralized compute configuration path."""
        task_dir = self.create_task_folder(task_name, task_stamp=task_stamp)
        if compute is None:
            overrides = compute_overrides or {}
            config_kwargs = {
                "job_name": task_name,
                "log_dir": task_dir / "logs",
                **overrides,
            }
            compute = self.configure_compute(**config_kwargs)

        generic = self.build_generic_sbatch_header(compute)
        script_text = self.compose_sbatch_script(generic, task_body)
        script_path = self.write_sbatch_script(task_dir, script_name, script_text)
        submit_result = self.submit_sbatch(script_path, submit=submit)

        payload: Dict[str, object] = {
            "task_name": task_name,
            "task_dir": str(task_dir),
            "mode": mode,
            "script_path": str(script_path),
            "job_id": submit_result["job_id"],
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "compute": asdict(compute),
        }
        if metadata_extra:
            payload.update(metadata_extra)

        self._persist_submission(task_dir, payload)
        return payload

    def submit_python_script(
        self,
        task_name: str,
        python_script: str,
        script_args: Optional[List[str]] = None,
        *,
        task_stamp: Optional[str] = None,
        submit: bool = True,
        nodes: int = 1,
        cpus_per_task: int = 1,
        mem_per_cpu_gb: float = 1.0,
        wall_time: str = "00:05:00",
        partition: Optional[str] = None,
        qos: Optional[str] = None,
        account: Optional[str] = None,
        modules: Optional[List[str]] = None,
        conda_env: Optional[str] = None,
        mail_type: Optional[str] = None,
        mail_user: Optional[str] = None,
        extra_sbatch_lines: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        body = self.build_python_task_body(python_script, script_args=script_args)
        return self.submit_task_body(
            task_name=task_name,
            script_name="run_python.sbatch",
            task_body=body,
            task_stamp=task_stamp,
            submit=submit,
            compute_overrides={
                "job_name": task_name,
                "nodes": nodes,
                "cpus_per_task": cpus_per_task,
                "mem_per_cpu_gb": mem_per_cpu_gb,
                "wall_time": wall_time,
                "partition": partition,
                "qos": qos,
                "account": account,
                "modules": modules,
                "conda_env": conda_env,
                "mail_type": mail_type,
                "mail_user": mail_user,
                "extra_sbatch_lines": extra_sbatch_lines,
            },
            mode="single_python",
        )

    def submit_index_range_python(
        self,
        task_name: str,
        worker_script: str,
        config_path: str,
        total_items: int,
        *,
        chunk_size: int = 5000,
        task_stamp: Optional[str] = None,
        submit: bool = True,
        nodes: int = 1,
        cpus_per_task: int = 1,
        mem_per_cpu_gb: float = 1.0,
        wall_time: str = "00:05:00",
        partition: Optional[str] = None,
        qos: Optional[str] = None,
        account: Optional[str] = None,
        modules: Optional[List[str]] = None,
        conda_env: Optional[str] = None,
        mail_type: Optional[str] = None,
        mail_user: Optional[str] = None,
        extra_sbatch_lines: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        if total_items <= 0:
            raise ValueError("total_items must be > 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        n_tasks = math.ceil(total_items / chunk_size)
        array_spec = f"0-{n_tasks - 1}"
        body = self.build_index_range_task_body(worker_script, config_path, total_items, chunk_size)
        return self.submit_task_body(
            task_name=task_name,
            script_name="run_index_range.sbatch",
            task_body=body,
            task_stamp=task_stamp,
            submit=submit,
            compute_overrides={
                "job_name": task_name,
                "array_spec": array_spec,
                "nodes": nodes,
                "cpus_per_task": cpus_per_task,
                "mem_per_cpu_gb": mem_per_cpu_gb,
                "wall_time": wall_time,
                "partition": partition,
                "qos": qos,
                "account": account,
                "modules": modules,
                "conda_env": conda_env,
                "mail_type": mail_type,
                "mail_user": mail_user,
                "extra_sbatch_lines": extra_sbatch_lines,
            },
            mode="index_range_python",
            metadata_extra={"array_spec": array_spec, "chunk_size": chunk_size, "total_items": total_items},
        )

    def submit_args_file_python_array(
        self,
        task_name: str,
        worker_script: str,
        args_file: str,
        *,
        total_items: Optional[int] = None,
        task_stamp: Optional[str] = None,
        submit: bool = True,
        nodes: int = 1,
        cpus_per_task: int = 1,
        mem_per_cpu_gb: float = 1.0,
        wall_time: str = "00:05:00",
        partition: Optional[str] = None,
        qos: Optional[str] = None,
        account: Optional[str] = None,
        modules: Optional[List[str]] = None,
        conda_env: Optional[str] = None,
        mail_type: Optional[str] = None,
        mail_user: Optional[str] = None,
        extra_sbatch_lines: Optional[List[str]] = None,
        line_argument_name: str = "--arg-json",
        worker_extra_args: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        args_path = Path(args_file)
        if total_items is None:
            total_items = len([line for line in args_path.read_text().splitlines() if line.strip()])
        if total_items <= 0:
            raise ValueError("total_items must be > 0")

        array_spec = f"0-{total_items - 1}"
        body = self.build_args_file_line_task_body(
            worker_script,
            str(args_path),
            line_argument_name=line_argument_name,
            worker_extra_args=worker_extra_args,
        )
        return self.submit_task_body(
            task_name=task_name,
            script_name="run_args_file_array.sbatch",
            task_body=body,
            task_stamp=task_stamp,
            submit=submit,
            compute_overrides={
                "job_name": task_name,
                "array_spec": array_spec,
                "nodes": nodes,
                "cpus_per_task": cpus_per_task,
                "mem_per_cpu_gb": mem_per_cpu_gb,
                "wall_time": wall_time,
                "partition": partition,
                "qos": qos,
                "account": account,
                "modules": modules,
                "conda_env": conda_env,
                "mail_type": mail_type,
                "mail_user": mail_user,
                "extra_sbatch_lines": extra_sbatch_lines,
            },
            mode="args_file_python_array",
            metadata_extra={"array_spec": array_spec, "total_items": total_items, "args_file": str(args_path)},
        )

    def submit_index_range_test_unit(
        self,
        task_name: str,
        worker_script: str,
        config_path: str,
        *,
        unit_start: int = 0,
        unit_end: int = 1,
        task_stamp: Optional[str] = None,
        submit: bool = True,
        nodes: int = 1,
        cpus_per_task: int = 1,
        mem_per_cpu_gb: float = 1.0,
        wall_time: str = "00:05:00",
        partition: Optional[str] = None,
        modules: Optional[List[str]] = None,
        conda_env: Optional[str] = None,
    ) -> Dict[str, object]:
        if unit_end <= unit_start:
            raise ValueError("unit_end must be greater than unit_start")

        body = self.build_single_index_range_body(worker_script, config_path, unit_start, unit_end)
        return self.submit_task_body(
            task_name=task_name,
            script_name="run_index_range_unit.sbatch",
            task_body=body,
            task_stamp=task_stamp,
            submit=submit,
            compute_overrides={
                "job_name": f"{task_name}_unit",
                "nodes": nodes,
                "cpus_per_task": cpus_per_task,
                "mem_per_cpu_gb": mem_per_cpu_gb,
                "wall_time": wall_time,
                "partition": partition,
                "modules": modules,
                "conda_env": conda_env,
            },
            mode="index_range_test_unit",
            metadata_extra={"unit_start": unit_start, "unit_end": unit_end},
        )

    @staticmethod
    def _load_job_ids(task_dir: Path) -> List[str]:
        metadata_path = task_dir / "submission.json"
        if not metadata_path.exists():
            return []
        data = json.loads(metadata_path.read_text())
        ids: List[str] = []
        for rec in data.get("jobs", []):
            job_id = rec.get("job_id")
            if job_id:
                ids.append(str(job_id))
        return ids

    def _find_error_logs(self, task_dir: Path) -> List[Dict[str, str]]:
        log_dir = task_dir / "logs"
        failures: List[Dict[str, str]] = []
        if not log_dir.exists():
            return failures
        for err_file in sorted(log_dir.glob("*.err")):
            if not self._err_file_has_error(err_file):
                continue
            job_id = err_file.stem.split("_", 1)[0]
            failures.append(
                {
                    "job_id": job_id,
                    "state": "LOG_ERROR",
                    "log_file": str(err_file),
                }
            )
        return failures

    def collect_task_status(self, task_dir: str, include_log_errors: bool = True) -> Dict[str, object]:
        """Collect running/complete/failed info for submitted jobs under task_dir."""
        resolved_task_dir = Path(task_dir)
        job_ids = self._load_job_ids(resolved_task_dir)
        status = {"running": [], "complete": [], "failed": [], "unknown": []}
        if not job_ids:
            return {"task_dir": str(resolved_task_dir), "summary": status, "raw": []}

        joined = ",".join(job_ids)
        sacct_result = self._run_command(["sacct", "-n", "-X", "-P", "-j", joined, "--format=JobIDRaw,State"])
        raw_rows: List[str] = []

        if sacct_result.returncode == 0 and sacct_result.stdout.strip():
            raw_rows = [line.strip() for line in sacct_result.stdout.splitlines() if line.strip()]
            seen = set()
            for row in raw_rows:
                parts = row.split("|")
                if len(parts) < 2:
                    continue
                jid = parts[0].strip()
                state = parts[1].strip()
                marker = (jid, state)
                if marker in seen:
                    continue
                seen.add(marker)
                bucket = self._slurm_state_bucket(state)
                status[bucket].append({"job_id": jid, "state": state})
        else:
            squeue_result = self._run_command(["squeue", "-h", "-j", joined, "-o", "%A|%T"])
            if squeue_result.returncode == 0 and squeue_result.stdout.strip():
                raw_rows = [line.strip() for line in squeue_result.stdout.splitlines() if line.strip()]
                for row in raw_rows:
                    parts = row.split("|")
                    if len(parts) < 2:
                        continue
                    jid = parts[0].strip()
                    state = parts[1].strip()
                    bucket = self._slurm_state_bucket(state)
                    status[bucket].append({"job_id": jid, "state": state})
                known = {entry["job_id"] for values in status.values() for entry in values}
                for jid in job_ids:
                    if jid not in known:
                        status["unknown"].append({"job_id": jid, "state": "UNKNOWN"})
            else:
                for jid in job_ids:
                    status["unknown"].append({"job_id": jid, "state": "UNAVAILABLE"})

        if include_log_errors:
            log_failures = self._find_error_logs(resolved_task_dir)
            existing = {(item["job_id"], item["state"]) for item in status["failed"]}
            for item in log_failures:
                marker = (item["job_id"], item["state"])
                if marker not in existing:
                    status["failed"].append(item)

        return {"task_dir": str(resolved_task_dir), "summary": status, "raw": raw_rows}

    def find_failed_jobs(self, task_dir: str) -> List[Dict[str, str]]:
        """Return failed SLURM jobs associated with task_dir."""
        status = self.collect_task_status(task_dir, include_log_errors=True)
        return status["summary"]["failed"]

    def cleanup_error_logs_if_no_error(self, task_dir: str) -> List[str]:
        """Delete .err log files only if no task failures and all .err logs are empty."""
        resolved_task_dir = Path(task_dir)
        failed = self.find_failed_jobs(str(resolved_task_dir))
        if failed:
            return []

        log_dir = resolved_task_dir / "logs"
        if not log_dir.exists():
            return []

        err_files = sorted(log_dir.glob("*.err"))
        if not err_files:
            return []

        if any(path.read_text().strip() for path in err_files):
            return []

        removed: List[str] = []
        for path in err_files:
            path.unlink(missing_ok=True)
            removed.append(str(path))
        return removed


def write_sbatch(
    job_name: str,
    worker_script: str,
    config_path: str,
    conda_env: str,
    total_items: int,
    chunk_size: int = 5000,
    nodes: int = 1,
    cpus_per_task: int = 1,
    mem_per_cpu: int = 8,
    wall_time: str = "00:15:00",
    partition: Optional[str] = None,
    log_dir: str = "logs",
    modules: Optional[List[str]] = None,
    mail_type: Optional[str] = "all",
    mail_user: Optional[str] = None,
    outfile: Optional[str] = None,
) -> Path:
    """Backward-compatible wrapper for creating an index-range sbatch script."""
    if total_items <= 0:
        raise ValueError("total_items must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    client = SlurmJobClient(root_folder=Path(log_dir).parent, default_conda_env=conda_env)
    n_tasks = math.ceil(total_items / chunk_size)
    compute = client.configure_compute(
        job_name=job_name,
        log_dir=Path(log_dir),
        nodes=nodes,
        cpus_per_task=cpus_per_task,
        mem_per_cpu_gb=mem_per_cpu,
        wall_time=wall_time,
        partition=partition,
        array_spec=f"0-{n_tasks - 1}",
        mail_type=mail_type,
        mail_user=mail_user,
        modules=modules,
        conda_env=conda_env,
    )
    body = client.build_index_range_task_body(worker_script, config_path, total_items, chunk_size)
    script = client.compose_sbatch_script(client.build_generic_sbatch_header(compute), body)

    sbatch_path = Path(outfile or f"{job_name}.sbatch")
    sbatch_path.write_text(script)
    print(f"Wrote {sbatch_path.resolve()}")
    return sbatch_path
