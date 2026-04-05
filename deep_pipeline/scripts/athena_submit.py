#!/usr/bin/env python3
"""Submit experiments to the Technion Athena GPU cluster via SLURM.

Usage:
    # Submit a single experiment
    python scripts/athena_submit.py --config configs/aki_v5_cross3.json --name aki_seed42

    # Submit with job chaining (for >24h runs)
    python scripts/athena_submit.py --config configs/aki_v5_cross3.json --chain 2

    # Submit multi-seed array job
    python scripts/athena_submit.py --config configs/aki_v5_cross3.json --seeds 42,7,2024

    # Override QoS (default: auto-selected)
    python scripts/athena_submit.py --config configs/aki_v5_cross3.json --qos 4d_1g

    # Check status of Athena jobs
    python scripts/athena_submit.py --status

    # Collect results from Athena
    python scripts/athena_submit.py --collect NAME

    # Sync code to Athena (shortcut for athena_sync.sh code)
    python scripts/athena_submit.py --sync
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

REPO = Path(__file__).resolve().parent.parent
QUEUE_PATH = REPO / "experiments" / "queue.yaml"
TEMPLATE_PATH = REPO / "scripts" / "athena_job.sh"
ATHENA_CONFIGS_DIR = REPO / "experiments" / ".athena_configs"
ATHENA_SCRIPTS_DIR = REPO / "experiments" / ".athena_scripts"

ATHENA_HOST = "omer.gotfrid@athena-login"
ATHENA_REPO = "~/Thesis/EHR_Translator/deep_pipeline"
ATHENA_LOG_DIR = "experiments/logs"  # relative to ATHENA_REPO (SBATCH resolves from submit dir)

# SLURM account (confirmed 2026-03-25)
ATHENA_ACCOUNT = "aran_prj"

# Path mapping: local → Athena (must use absolute path, not ~)
PATH_MAPPINGS = {
    "/bigdata/omerg/Thesis": "/home/omer.gotfrid/Thesis",
}

# SSH options
SSH_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
]

# QoS tiers with their properties
QOS_TIERS = {
    "2h_2g":  {"wall": "02:00:00",   "priority": 1000, "max_jobs": 3, "max_gpu": 2},
    "12h_4g": {"wall": "12:00:00",   "priority": 500,  "max_jobs": 3, "max_gpu": 4},
    "24h_1g": {"wall": "1-00:00:00", "priority": 300,  "max_jobs": 4, "max_gpu": 1},
    "24h_4g": {"wall": "1-00:00:00", "priority": 250,  "max_jobs": 2, "max_gpu": 4},
    "4d_1g":  {"wall": "4-00:00:00", "priority": 50,   "max_jobs": 8, "max_gpu": 1},
    "72h_8g": {"wall": "3-00:00:00", "priority": 50,   "max_jobs": 1, "max_gpu": 8},
}

# Partition → conda env mapping. Ordered by preference (idle nodes first).
# Blackwell/H200 partitions use yaib-cu128 (different PyTorch/CUDA build),
# producing non-comparable results. They are excluded from defaults but can
# be requested explicitly via --partition.
DEFAULT_PARTITIONS = {"l40s-shared", "a100-public"}

PARTITIONS = {
    "rtx6k-shared": {
        "conda_env": "yaib-cu128",
        "total_gpus": 16,
        "compatible_qos": ["2h_2g", "12h_4g", "24h_4g", "4d_1g", "72h_8g"],
        "priority": 1,
    },
    "h200-shared": {
        "conda_env": "yaib-cu128",
        "total_gpus": 8,
        "compatible_qos": ["2h_2g", "12h_4g", "24h_1g", "24h_4g", "4d_1g", "72h_8g"],
        "priority": 2,
    },
    "l40s-shared": {
        "conda_env": "yaib",
        "total_gpus": 16,
        "compatible_qos": ["2h_2g", "12h_4g", "24h_1g", "24h_4g", "4d_1g", "72h_8g"],
        "priority": 3,
    },
    "a100-public": {
        "conda_env": "yaib",
        "total_gpus": 8,
        "compatible_qos": ["2h_2g", "12h_4g", "24h_1g", "24h_4g", "4d_1g", "72h_8g"],
        "priority": 4,
    },
}


# ---------------------------------------------------------------------------
# Config remapping
# ---------------------------------------------------------------------------

def remap_config(config_path: str) -> str:
    """Read a config JSON and remap all paths for Athena.
    Returns the remapped JSON string."""
    with open(config_path) as f:
        content = f.read()
    for local_prefix, remote_prefix in PATH_MAPPINGS.items():
        content = content.replace(local_prefix, remote_prefix)
    return content


def infer_task(config_path: str) -> str:
    """Infer task name from config filename."""
    name = Path(config_path).stem.lower()
    for task in ["mortality", "aki", "sepsis", "los", "kf", "kidney_function"]:
        if task in name:
            return task
    return "unknown"


# ---------------------------------------------------------------------------
# Partition selection
# ---------------------------------------------------------------------------

def _query_idle_gpus(partitions: list[str]) -> dict[str, int]:
    """Count free GPUs per partition = total - currently allocated."""
    idle_counts = {}
    for part_name in partitions:
        total = PARTITIONS[part_name]["total_gpus"]
        result = ssh_run(
            f"squeue -p {part_name} --noheader -o '%b' 2>/dev/null",
            timeout=10,
        )
        used = 0
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line and 'gpu' in line.lower():
                    try:
                        used += int(line.split(':')[-1])
                    except ValueError:
                        used += 1
        idle_counts[part_name] = max(0, total - used)
    return idle_counts


def select_partitions(qos: str) -> str:
    """Return comma-separated list of default-safe partitions compatible with this QoS.
    Only partitions in DEFAULT_PARTITIONS are included (yaib cu118 env).
    SLURM will schedule on whichever has a free slot first.
    The job script auto-detects GPU type and activates the right conda env."""
    compatible = [
        name for name, info in
        sorted(PARTITIONS.items(), key=lambda x: x[1]["priority"])
        if qos in info["compatible_qos"] and name in DEFAULT_PARTITIONS
    ]
    if not compatible:
        logging.warning(f"No partition supports QoS '{qos}', falling back to l40s-shared")
        return "l40s-shared"

    result = ",".join(compatible)
    logging.info(f"Targeting partitions: {result} (SLURM picks first available)")
    return result


def adapt_qos_for_partition(qos: str, partition: str) -> str:
    """If QoS is incompatible with partition, find closest compatible alternative."""
    info = PARTITIONS.get(partition)
    if not info or qos in info["compatible_qos"]:
        return qos
    fallback = {"24h_1g": "24h_4g"}.get(qos)
    if fallback and fallback in info["compatible_qos"]:
        logging.info(f"QoS '{qos}' incompatible with {partition}, using '{fallback}'")
        return fallback
    for compat in info["compatible_qos"]:
        if QOS_TIERS.get(compat, {}).get("wall") == QOS_TIERS.get(qos, {}).get("wall"):
            return compat
    return qos


# ---------------------------------------------------------------------------
# QoS selection
# ---------------------------------------------------------------------------

def select_qos(config_path: str, name: str = "", chain: int = 1) -> tuple[str, str]:
    """Auto-select QoS and wall time based on experiment type.
    Returns (qos_name, wall_time)."""
    fname = Path(config_path).stem.lower()

    # Bootstrap CI / temp scaling — short jobs
    if "bootstrap" in name or "tempscale" in name or "bootstrap" in fname or "tempscale" in fname:
        return "2h_2g", QOS_TIERS["2h_2g"]["wall"]

    # Read config to check translator type and epochs
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return "24h_1g", QOS_TIERS["24h_1g"]["wall"]

    ttype = cfg.get("translator", {}).get("type", "")
    epochs = cfg.get("training", {}).get("epochs", 50)

    # DA baselines (fast, no pretrain)
    if ttype in ("dann", "coral", "codats", "cluda", "raincoat", "acon", "stats_only",
                 "e2e_cluda", "e2e_raincoat", "e2e_acon",
                 "e2e_dann", "e2e_coral", "e2e_codats"):
        if epochs <= 30:
            return "12h_4g", QOS_TIERS["12h_4g"]["wall"]
        return "24h_1g", QOS_TIERS["24h_1g"]["wall"]

    # Short runs (screening, few epochs)
    if epochs <= 15:
        return "12h_4g", QOS_TIERS["12h_4g"]["wall"]

    # Explicit chaining → user expects >24h, use 4d_1g for each segment
    if chain > 1:
        return "4d_1g", QOS_TIERS["4d_1g"]["wall"]

    # Default: 24h single-GPU (fits most 50-epoch runs on A100)
    return "24h_1g", QOS_TIERS["24h_1g"]["wall"]


# ---------------------------------------------------------------------------
# SLURM script generation
# ---------------------------------------------------------------------------

def generate_sbatch_script(
    name: str,
    config_path_athena: str,
    output_path_athena: str,
    qos: str,
    wall_time: str,
    account: str = "",
    partitions: str = "",
    command: str = "train_and_eval",
) -> str:
    """Generate an sbatch script from the template."""
    template = TEMPLATE_PATH.read_text()
    script = template
    script = script.replace("__EXPNAME__", name)
    script = script.replace("__WALLTIME__", wall_time)
    script = script.replace("__QOS__", qos)
    script = script.replace("__CONFIGPATH__", config_path_athena)
    script = script.replace("__OUTPUTPATH__", output_path_athena)
    script = script.replace("__ACCOUNT__", account or ATHENA_ACCOUNT)
    script = script.replace("__PARTITIONS__", partitions or "l40s-shared")
    script = script.replace("__COMMAND__", command)
    return script


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def ssh_run(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command on Athena via SSH."""
    full_cmd = ["ssh"] + SSH_OPTS + [ATHENA_HOST, cmd]
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)


def rsync_to_athena(local_path: str, remote_path: str) -> bool:
    """rsync a file to Athena."""
    result = subprocess.run(
        ["rsync", "-az", local_path, f"{ATHENA_HOST}:{remote_path}"],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0


def rsync_from_athena(remote_path: str, local_path: str) -> bool:
    """rsync a file from Athena."""
    result = subprocess.run(
        ["rsync", "-az", f"{ATHENA_HOST}:{remote_path}", local_path],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

def load_queue() -> dict:
    """Load the experiment queue."""
    with open(QUEUE_PATH) as f:
        return yaml.safe_load(f) or {}


def save_queue(queue: dict):
    """Save the experiment queue."""
    with open(QUEUE_PATH, "w") as f:
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False, width=200)


def add_queue_entry(name: str, config: str, output: str, notes: str = "",
                    slurm_job_id: str = "", chain_ids: list[str] | None = None):
    """Add an experiment entry to queue.yaml (or update existing by name)."""
    queue = load_queue()
    experiments = queue.setdefault("experiments", [])

    # Update existing entry if name matches (avoid duplicates)
    for exp in experiments:
        if exp.get("name") == name:
            exp["status"] = "athena_pending"
            exp["server"] = "athena"
            if slurm_job_id:
                exp["slurm_job_id"] = slurm_job_id
            if chain_ids:
                exp["slurm_chain_ids"] = chain_ids
            exp["submitted"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_queue(queue)
            return

    entry = {
        "name": name,
        "config": config,
        "output": output,
        "status": "athena_pending",
        "server": "athena",
        "notes": notes,
        "submitted": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if slurm_job_id:
        entry["slurm_job_id"] = slurm_job_id
    if chain_ids:
        entry["slurm_chain_ids"] = chain_ids

    experiments.append(entry)
    save_queue(queue)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def submit_experiment(
    config_path: str,
    name: str | None = None,
    qos: str | None = None,
    chain: int = 1,
    output: str | None = None,
    notes: str = "",
    dry_run: bool = False,
    account: str = "",
    partition: str = "",
    command: str = "train_and_eval",
) -> list[str]:
    """Submit a single experiment to Athena. Returns list of SLURM job IDs."""

    config_path = str(Path(config_path))
    if not Path(config_path).exists():
        logging.error(f"Config not found: {config_path}")
        return []

    # Derive name from config if not provided
    if not name:
        name = Path(config_path).stem

    # Auto-select QoS if not specified
    if not qos:
        qos, wall_time = select_qos(config_path, name, chain)
    else:
        wall_time = QOS_TIERS.get(qos, {}).get("wall", "1-00:00:00")

    # Auto-select partitions if not specified
    # SLURM accepts comma-separated partitions and picks whichever has a slot first.
    # The job script auto-detects GPU type and activates the right conda env.
    if not partition or partition == "auto":
        # Find a QoS that works on the most partitions
        # Some QoS (e.g. 24h_1g) are incompatible with certain partitions.
        # Try the original QoS first; if it's only compatible with a few, check
        # if adapting it unlocks more partitions.
        partitions_str = select_partitions(qos)
        n_orig = len(partitions_str.split(","))
        adapted = adapt_qos_for_partition(qos, "rtx6k-shared")  # test adaptation
        partitions_adapted = select_partitions(adapted)
        n_adapted = len(partitions_adapted.split(","))
        if n_adapted > n_orig and adapted != qos:
            logging.info(f"Adapting QoS '{qos}' → '{adapted}' to unlock more partitions ({n_orig} → {n_adapted})")
            qos = adapted
            wall_time = QOS_TIERS.get(qos, {}).get("wall", wall_time)
            partitions_str = partitions_adapted
    else:
        partitions_str = partition

    logging.info(f"Experiment: {name}")
    logging.info(f"Partitions: {partitions_str}")
    logging.info(f"QoS: {qos} (wall: {wall_time}, priority: {QOS_TIERS.get(qos, {}).get('priority', '?')})")

    # Remap config paths for Athena
    remapped = remap_config(config_path)
    ATHENA_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    local_config = ATHENA_CONFIGS_DIR / f"{name}.json"
    local_config.write_text(remapped)

    # Derive output path
    if not output:
        task = infer_task(config_path)
        output = f"runs/{name}/eval_{task}.parquet"

    # Remap output path
    athena_config = f"{ATHENA_REPO}/experiments/.athena_configs/{name}.json"
    athena_output = f"{ATHENA_REPO}/{output}"

    # Generate sbatch script
    script = generate_sbatch_script(name, athena_config, athena_output, qos, wall_time,
                                    account=account, partitions=partitions_str, command=command)
    ATHENA_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    local_script = ATHENA_SCRIPTS_DIR / f"{name}.sh"
    local_script.write_text(script)

    if dry_run:
        logging.info(f"[DRY RUN] Would submit: {local_script}")
        logging.info(f"  Config: {athena_config}")
        logging.info(f"  Output: {athena_output}")
        logging.info(f"  Chain: {chain} jobs")
        return []

    # Upload config and script to Athena
    logging.info("Uploading config and script to Athena...")
    ssh_run(f"mkdir -p {ATHENA_REPO}/experiments/.athena_configs {ATHENA_REPO}/experiments/.athena_scripts {ATHENA_REPO}/{ATHENA_LOG_DIR}")
    if not rsync_to_athena(str(local_config), athena_config):
        logging.error("Failed to upload config to Athena")
        return []
    athena_script = f"{ATHENA_REPO}/experiments/.athena_scripts/{name}.sh"
    if not rsync_to_athena(str(local_script), athena_script):
        logging.error("Failed to upload script to Athena")
        return []

    # Submit job chain (cd into repo so SLURM resolves relative --output/--error paths)
    job_ids = []
    prev_id = None
    for i in range(chain):
        dep = f"--dependency=afterany:{prev_id}" if prev_id else ""
        cmd = f"cd {ATHENA_REPO} && sbatch --parsable {dep} {athena_script}"
        logging.info(f"Submitting job {i+1}/{chain}: {cmd}")
        result = ssh_run(cmd, timeout=30)
        if result.returncode != 0:
            logging.error(f"sbatch failed: {result.stderr.strip()}")
            break
        job_id = result.stdout.strip()
        job_ids.append(job_id)
        prev_id = job_id
        logging.info(f"  Job ID: {job_id}")

    # Record in queue
    if job_ids:
        add_queue_entry(
            name=name,
            config=config_path,
            output=output,
            notes=notes or f"Athena {qos}, chain={chain}",
            slurm_job_id=job_ids[0],
            chain_ids=job_ids if len(job_ids) > 1 else None,
        )
        logging.info(f"Submitted {len(job_ids)} job(s): {', '.join(job_ids)}")

    return job_ids


def submit_multiseed(
    config_path: str,
    seeds: list[int],
    qos: str | None = None,
    chain: int = 1,
    dry_run: bool = False,
    account: str = "",
    partition: str = "",
):
    """Submit multiple seeds as separate experiments."""
    base_name = Path(config_path).stem

    for seed in seeds:
        # Load config, override seed and run_dir
        with open(config_path) as f:
            cfg = json.load(f)

        cfg.setdefault("training", {})["training_seed"] = seed

        # Update run_dir to include seed
        if "run_dir" in cfg.get("training", {}):
            rd = cfg["training"]["run_dir"]
            # Append seed suffix
            cfg["training"]["run_dir"] = f"{rd}_seed{seed}"

        # Write seed-specific config
        seed_name = f"{base_name}_seed{seed}"
        seed_config = ATHENA_CONFIGS_DIR / f"{seed_name}_local.json"
        ATHENA_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(seed_config, "w") as f:
            json.dump(cfg, f, indent=2)

        submit_experiment(
            config_path=str(seed_config),
            name=seed_name,
            qos=qos,
            chain=chain,
            notes=f"Multi-seed: seed={seed}",
            dry_run=dry_run,
            account=account,
            partition=partition,
        )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def show_status():
    """Show status of Athena SLURM jobs."""
    # Check SLURM queue
    logging.info("Querying Athena SLURM queue...")
    result = ssh_run("squeue -u omer.gotfrid -o '%.12i %.30j %.8T %.10M %.6D %.4C %.8m %R' 2>/dev/null", timeout=15)
    if result.returncode == 0 and result.stdout.strip():
        print("\n=== Active SLURM Jobs ===")
        print(result.stdout)
    else:
        print("\nNo active SLURM jobs on Athena.")

    # Check queue.yaml for Athena entries
    queue = load_queue()
    athena_exps = [e for e in queue.get("experiments", [])
                   if e.get("server") == "athena"]
    if athena_exps:
        print(f"\n=== Athena Experiments in Queue ({len(athena_exps)} total) ===")
        for exp in athena_exps:
            status = exp.get("status", "?")
            name = exp.get("name", "?")
            job_id = exp.get("slurm_job_id", "?")
            submitted = exp.get("submitted", "?")
            results = exp.get("results", {})
            result_str = f" → {results}" if results else ""
            print(f"  [{status:>15}] {name} (job {job_id}, submitted {submitted}){result_str}")

    # Also check recent completed jobs via sacct
    result = ssh_run(
        "sacct -u omer.gotfrid --starttime=$(date -d '7 days ago' +%Y-%m-%d) "
        "-o JobID,JobName%30,State,Elapsed,MaxRSS,ExitCode --noheader 2>/dev/null",
        timeout=15,
    )
    if result.returncode == 0 and result.stdout.strip():
        print("\n=== Recent SLURM Job History (7 days) ===")
        print(result.stdout)


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def collect_results(name: str):
    """Collect results from Athena: run directory (checkpoints, predictions, logs) + SLURM logs."""
    queue = load_queue()
    exp = None
    for e in queue.get("experiments", []):
        if e.get("name") == name:
            exp = e
            break

    if not exp:
        logging.error(f"Experiment '{name}' not found in queue")
        return

    # Determine run_dir from config or output path
    config_path = exp.get("config", "")
    run_dir = None
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            run_dir = cfg.get("output", {}).get("run_dir")
        except Exception:
            pass
    if not run_dir:
        output = exp.get("output", "")
        if "runs/" in output:
            run_dir = str(Path(output).parent)
        else:
            run_dir = f"runs/{name}"

    # Collect run directory (eval parquets, predictions, checkpoints, plots, logs)
    remote_run = f"/home/omer.gotfrid/Thesis/EHR_Translator/deep_pipeline/{run_dir}/"
    local_run = REPO / run_dir
    local_run.mkdir(parents=True, exist_ok=True)
    logging.info(f"Collecting run directory: {remote_run} → {local_run}/")
    rsync_from_athena(remote_run, str(local_run) + "/")

    # Collect SLURM stdout/stderr logs
    local_log_dir = REPO / "experiments" / "logs"
    local_log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", f"{ATHENA_HOST}:/home/omer.gotfrid/Thesis/EHR_Translator/deep_pipeline/experiments/logs/athena_ehr_{name}*",
         str(local_log_dir) + "/"],
        capture_output=True, text=True, timeout=60,
    )
    # Also try with truncated name (SLURM truncates long job names)
    short_name = name[:20]
    if short_name != name:
        subprocess.run(
            ["rsync", "-az", f"{ATHENA_HOST}:/home/omer.gotfrid/Thesis/EHR_Translator/deep_pipeline/experiments/logs/athena_ehr_{short_name}*",
             str(local_log_dir) + "/"],
            capture_output=True, text=True, timeout=60,
        )

    # Try to run collect_result.py
    if task != "unknown":
        try:
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "collect_result.py"), name, task],
                cwd=str(REPO), timeout=30, capture_output=True,
            )
            result_path = REPO / "experiments" / "results" / f"{name}_{task}.json"
            if result_path.exists():
                data = json.loads(result_path.read_text())
                diff = data.get("difference", {})
                if diff:
                    exp["results"] = diff
                    exp["status"] = "athena_done"
                    save_queue(queue)
                    logging.info(f"Results: {diff}")
        except Exception as e:
            logging.warning(f"collect_result.py failed: {e}")


# ---------------------------------------------------------------------------
# Sync shortcut
# ---------------------------------------------------------------------------

def sync_code():
    """Run athena_sync.sh code."""
    sync_script = REPO / "scripts" / "athena_sync.sh"
    subprocess.run(["bash", str(sync_script), "code"], cwd=str(REPO))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Submit experiments to Athena GPU cluster")
    parser.add_argument("--config", help="Path to experiment config JSON")
    parser.add_argument("--name", help="Experiment name (default: config filename)")
    parser.add_argument("--qos", choices=list(QOS_TIERS.keys()), help="SLURM QoS (default: auto)")
    parser.add_argument("--account", default="", help=f"SLURM account (default: {ATHENA_ACCOUNT})")
    parser.add_argument("--partition", default="auto", help="SLURM partition (default: auto-select most idle)")
    parser.add_argument("--chain", type=int, default=1, help="Number of chained jobs (for >24h runs)")
    parser.add_argument("--seeds", help="Comma-separated seeds for multi-seed submission")
    parser.add_argument("--output", help="Output parquet path (default: auto)")
    parser.add_argument("--notes", default="", help="Notes for queue entry")
    parser.add_argument("--command", default="train_and_eval", help="run.py subcommand (default: train_and_eval)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted")
    parser.add_argument("--status", action="store_true", help="Show Athena job status")
    parser.add_argument("--collect", metavar="NAME", help="Collect results for experiment")
    parser.add_argument("--sync", action="store_true", help="Sync code to Athena")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.collect:
        collect_results(args.collect)
        return

    if args.sync:
        sync_code()
        return

    if not args.config:
        parser.error("--config is required for submission")

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        submit_multiseed(
            config_path=args.config,
            seeds=seeds,
            qos=args.qos,
            chain=args.chain,
            dry_run=args.dry_run,
            account=args.account,
            partition=args.partition,
        )
    else:
        submit_experiment(
            config_path=args.config,
            name=args.name,
            qos=args.qos,
            chain=args.chain,
            output=args.output,
            notes=args.notes,
            dry_run=args.dry_run,
            account=args.account,
            partition=args.partition,
            command=args.command,
        )


if __name__ == "__main__":
    main()
