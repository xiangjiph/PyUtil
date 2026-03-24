from pathlib import Path
import math
import textwrap
from typing import List, Optional

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
    """Generate an sbatch script that launches a SLURM job array.

    Parameters
    ----------
    job_name : str
        Name given to the SLURM job (#SBATCH -J).
    worker_script : str
        Path to the Python worker script that expects --start and --end CLI args.
    config_path : str
        Path to the YAML/JSON config file consumed by *worker_script*.
    conda_env : str
        Name of the Conda environment to activate.
    total_items : int
        Total number of 1‑second tasks to process (e.g. number of bounding boxes).
    chunk_size : int, default 5000
        How many items each array task should process.
    nodes : int, default 1
        Nodes per array task (#SBATCH -N).
    cpus_per_task : int, default 4
        Logical CPU cores per array task (#SBATCH -c).
    wall_time : str, default "00:15:00"
        Wall‑time limit per array task.
    partition : str | None
        SLURM partition/queue, if the cluster uses them.
    log_dir : str, default "logs"
        Directory for %A_%a.out stdout/stderr files. Created if absent.
    modules : list[str] | None
        Additional `module load` lines before Conda activation.
    outfile : str | None
        Path of the .sbatch file to write. Defaults to {job_name}.sbatch in CWD.

    Returns
    -------
    pathlib.Path
        Path to the generated sbatch file.
    """

    n_tasks = math.ceil(total_items / chunk_size)
    array_spec = f"0-{n_tasks - 1}"
    sbatch_path = Path(outfile or f"{job_name}.sbatch")

    # Ensure the log directory exists
    Path(log_dir).mkdir(exist_ok=True)

    module_lines = ""
    if modules:
        module_lines = "\n".join(f"module load {m}" for m in modules)

    partition_line = f"#SBATCH -p {partition}\n" if partition else ""

    script = textwrap.dedent(
        f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array={array_spec}
#SBATCH --nodes={nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}G
#SBATCH --time={wall_time}
#SBATCH --mail-type={mail_type}
#SBATCH --mail-user={mail_user}
{partition_line}#SBATCH --output={log_dir}/%A_%a.out


{module_lines}
source ~/.bashrc
conda activate {conda_env}

CHUNK={chunk_size}
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

python {worker_script} --start ${{START}} --end ${{END}} --cfg {config_path}
"""
            )

    sbatch_path.write_text(script)
    print(f"Wrote {sbatch_path.resolve()}")
    return sbatch_path


# --- Usage example --------------------------------------------------------
if __name__ == "__main__":
    write_sbatch(
        job_name="bbox",
        worker_script="worker.py",
        config_path="params.yml",
        conda_env="bbox-env",
        total_items=1_000_000,
        chunk_size=5000,
        wall_time="00:10:00",
    )
