#!/usr/bin/env python3
"""GPU Experiment Scheduler — reads experiments/queue.yaml, manages GPU assignment.

Supports multiple servers (local + remote via SSH) and branch-aware git worktrees.

Usage:
    python scripts/gpu_scheduler.py              # Start scheduler daemon
    python scripts/gpu_scheduler.py --status     # Show queue status
    python scripts/gpu_scheduler.py --dry-run    # Show what would launch
    python scripts/gpu_scheduler.py --add --name NAME --config PATH [--notes TEXT] [--server SERVER] [--branch BRANCH]
    python scripts/gpu_scheduler.py --cleanup [--branch BRANCH]  # Remove worktrees

Branch support: experiments with a 'branch' field run from git worktrees,
providing full code isolation. Checkpoints/logs are centralized in the main tree.
Designed to run in a tmux session. Graceful shutdown with Ctrl+C.
"""

import argparse
import fcntl
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
QUEUE_PATH = REPO / "experiments" / "queue.yaml"
QUEUE_LOCK_PATH = REPO / "experiments" / "queue.yaml.lock"
LOG_DIR = REPO / "experiments" / "logs"
RESULTS_DIR = REPO / "experiments" / "results"
SCHEDULER_LOG = LOG_DIR / "scheduler.log"

# Git worktree paths for branch-aware experiment isolation
GIT_ROOT = REPO.parent                          # EHR_Translator/
WORKTREE_BASE = GIT_ROOT.parent / "EHR_Translator_worktrees"
WORKTREE_CONFIGS_DIR = REPO / "experiments" / ".worktree_configs"

# Track running subprocesses for cleanup (local experiments only)
_running_procs: dict[str, subprocess.Popen] = {}
_shutdown = False

# SSH base options for connection pooling and non-interactive operation
_SSH_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/ssh-sched-%r@%h",
    "-o", "ControlPersist=300",
]

# SSH options WITHOUT ControlMaster — needed for commands that launch background
# processes, because ControlMaster keeps the connection alive via inherited FDs.
_SSH_OPTS_NO_CONTROL = [
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
]


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    name: str
    host: str | None  # None = local
    gpu_priority: list[int]
    day_max_gpus: int
    night_max_gpus: int
    repo_path: str = ""
    conda_env: str = "yaib"
    path_mappings: dict[str, str] = field(default_factory=dict)
    slurm: bool = False  # SLURM-managed servers are skipped by this scheduler

    @property
    def is_local(self) -> bool:
        return self.host is None


def _parse_servers(settings: dict) -> dict[str, ServerConfig]:
    """Parse servers from settings. Backward-compatible: no 'servers' key = single local."""
    servers_cfg = settings.get("servers")
    if not servers_cfg:
        # Legacy single-server mode
        return {
            "local": ServerConfig(
                name="local",
                host=None,
                gpu_priority=settings.get("gpu_priority", [0, 1, 2, 3]),
                day_max_gpus=settings.get("day_max_gpus", 2),
                night_max_gpus=settings.get("night_max_gpus", 3),
            )
        }

    servers = {}
    for name, cfg in servers_cfg.items():
        name = str(name)  # YAML may parse numeric keys (e.g. 3090) as int
        servers[name] = ServerConfig(
            name=name,
            host=cfg.get("host"),
            gpu_priority=cfg.get("gpu_priority", [0, 1, 2, 3]),
            day_max_gpus=cfg.get("day_max_gpus", 2),
            night_max_gpus=cfg.get("night_max_gpus", 2),
            repo_path=cfg.get("repo_path", ""),
            conda_env=cfg.get("conda_env", "yaib"),
            path_mappings=cfg.get("path_mappings", {}),
            slurm=cfg.get("slurm", False),
        )
    return servers


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [scheduler] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(SCHEDULER_LOG, mode="a"),
        ],
    )


# ---------------------------------------------------------------------------
# Queue I/O
# ---------------------------------------------------------------------------

@contextmanager
def _queue_lock():
    """Acquire an exclusive file lock on the queue lock file.
    Prevents race conditions between concurrent scheduler/add operations."""
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(QUEUE_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def load_queue() -> dict:
    if not QUEUE_PATH.exists():
        logging.error(f"Queue file not found: {QUEUE_PATH}")
        sys.exit(1)
    with open(QUEUE_PATH) as f:
        return yaml.safe_load(f)


def save_queue(queue: dict):
    """Atomic write: tmp file + rename to prevent corruption."""
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(QUEUE_PATH) + ".tmp"
    with open(tmp, "w") as f:
        # Write header comment manually, then dump YAML
        f.write("# ============================================================\n")
        f.write("# Experiment Queue — edit this file to manage experiments\n")
        f.write("# Reorder pending experiments by moving entries up/down\n")
        f.write("# The scheduler runs top-to-bottom through pending experiments\n")
        f.write("# ============================================================\n\n")
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)
    os.rename(tmp, str(QUEUE_PATH))


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _ssh_run(host: str, cmd: str, timeout: int = 30, use_control: bool = True) -> tuple[int, str]:
    """Run a command on a remote host via SSH. Returns (returncode, stdout)."""
    opts = _SSH_OPTS if use_control else _SSH_OPTS_NO_CONTROL
    try:
        result = subprocess.run(
            ["ssh"] + opts + [host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        logging.warning(f"SSH timeout ({timeout}s) to {host}")
        return 1, ""
    except Exception as e:
        logging.warning(f"SSH failed to {host}: {e}")
        return 1, ""


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _parse_nvidia_smi_output(output: str, threshold_mb: int) -> list[int]:
    """Parse nvidia-smi CSV output into list of free GPU indices."""
    free = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        try:
            idx, mem = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            continue
        if mem < threshold_mb:
            free.append(idx)
    return free


def get_free_gpus(threshold_mb: int, server: ServerConfig | None = None) -> list[int]:
    """Return GPU indices with memory usage below threshold (local or remote)."""
    if server is not None and server.slurm:
        return []  # SLURM servers managed by athena_submit.py, not this scheduler
    if server is not None and not server.is_local:
        return get_free_gpus_remote(server, threshold_mb)

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=10,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logging.warning(f"nvidia-smi failed: {e}")
        return []

    return _parse_nvidia_smi_output(output, threshold_mb)


def get_free_gpus_remote(server: ServerConfig, threshold_mb: int) -> list[int]:
    """Get free GPUs on a remote server via SSH."""
    rc, output = _ssh_run(
        server.host,
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
    )
    if rc != 0:
        logging.warning(f"Cannot query GPUs on {server.name} ({server.host})")
        return []
    return _parse_nvidia_smi_output(output, threshold_mb)


def get_max_gpus(settings_or_server) -> int:
    """Return max GPUs allowed based on time of day and day of week."""
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    is_weekend_day = weekday in (4, 5)  # Fri, Sat

    if isinstance(settings_or_server, ServerConfig):
        day_max = settings_or_server.day_max_gpus
        night_max = settings_or_server.night_max_gpus
    else:
        day_max = settings_or_server.get("day_max_gpus", 2)
        night_max = settings_or_server.get("night_max_gpus", 3)

    if 9 <= hour < 21 and not is_weekend_day:
        return day_max
    return night_max


def select_gpus(server: ServerConfig, free_gpus: list[int], running_gpus: set[int]) -> list[int]:
    """Select available GPUs on a server respecting priority and time limits."""
    max_gpus = get_max_gpus(server)
    priority = server.gpu_priority

    # GPUs that are free AND not running our experiments, in priority order
    available = [g for g in priority if g in free_gpus and g not in running_gpus]

    # How many more can we launch?
    slots = max(0, max_gpus - len(running_gpus))
    return available[:slots]


# ---------------------------------------------------------------------------
# Task inference
# ---------------------------------------------------------------------------

def infer_task(config_path: str) -> str:
    """Infer task name from config filename."""
    name = Path(config_path).stem.lower()
    for task in ["mortality", "aki", "sepsis", "los", "kf", "kidney_function"]:
        if task in name:
            return task
    return "unknown"


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def pid_is_alive(pid: int) -> bool:
    """Check if a local process with given PID is still running (not zombie)."""
    try:
        status_path = f"/proc/{pid}/status"
        if os.path.exists(status_path):
            with open(status_path) as f:
                for line in f:
                    if line.startswith("State:"):
                        # Z = zombie, X = dead
                        state = line.split()[1]
                        return state not in ("Z", "X")
        # Fallback: signal check
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def pid_is_alive_remote(server: ServerConfig, pid: int) -> bool:
    """Check if a process is alive on a remote server. Conservative: returns True on SSH failure.
    Falls back to checking child processes via pgrep -P when kill -0 fails."""
    rc, output = _ssh_run(
        server.host,
        f"kill -0 {pid} 2>/dev/null && echo ALIVE || "
        f"(pgrep -P {pid} >/dev/null 2>&1 && echo ALIVE || echo DEAD)",
    )
    if rc != 0:
        # SSH failure — assume alive (conservative)
        return True
    return output.strip() == "ALIVE"


def batch_pid_check_remote(server: ServerConfig, pids: list[int]) -> dict[int, bool]:
    """Check multiple PIDs on a remote server in a single SSH call.
    Falls back to pgrep -P for child process detection when kill -0 fails."""
    if not pids:
        return {}
    checks = "; ".join(
        f'kill -0 {pid} 2>/dev/null && echo "{pid} ALIVE" || '
        f'(pgrep -P {pid} >/dev/null 2>&1 && echo "{pid} ALIVE" || echo "{pid} DEAD")'
        for pid in pids
    )
    rc, output = _ssh_run(server.host, checks)
    if rc != 0:
        # SSH failure — assume all alive
        return {pid: True for pid in pids}
    results = {}
    for line in output.split("\n"):
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                results[int(parts[0])] = parts[1] == "ALIVE"
            except ValueError:
                pass
    # Any PIDs not in output: assume alive (conservative)
    for pid in pids:
        if pid not in results:
            results[pid] = True
    return results


def recover_stale(experiments: list[dict], servers: dict[str, ServerConfig]):
    """On startup, mark 'running' experiments with dead PIDs as failed."""
    # Group remote experiments by server for batch checking
    remote_groups: dict[str, list[dict]] = {}
    for exp in experiments:
        if exp.get("status") != "running":
            continue
        pid = exp.get("pid")
        if not pid:
            continue
        srv_name = exp.get("server", "local")
        server = servers.get(srv_name)
        if server and not server.is_local:
            remote_groups.setdefault(srv_name, []).append(exp)
        elif server and server.is_local:
            if not pid_is_alive(pid):
                output_path = REPO / exp.get("output", "")
                if output_path.exists():
                    logging.info(
                        f"Stale experiment '{exp['name']}' (PID {pid} dead) — "
                        f"output exists, marking done"
                    )
                    exp["status"] = "done"
                    exp["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    logging.warning(
                        f"Stale experiment '{exp['name']}' (PID {pid} dead), marking failed"
                    )
                    exp["status"] = "failed"
                    exp["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    exp["error"] = "Process died (scheduler restart recovery)"

    # Batch check remote PIDs
    for srv_name, exps in remote_groups.items():
        server = servers[srv_name]
        pids = [e["pid"] for e in exps]
        alive_map = batch_pid_check_remote(server, pids)
        for exp in exps:
            if not alive_map.get(exp["pid"], True):
                # Check if output parquet exists on the remote server
                remote_output = f"{server.repo_path}/{exp.get('output', '')}"
                rc_out, _ = _ssh_run(server.host, f"test -f {remote_output}")
                if rc_out == 0:
                    logging.info(
                        f"Stale remote experiment '{exp['name']}' on {srv_name} "
                        f"(PID {exp['pid']} dead) — output exists, marking done"
                    )
                    exp["status"] = "done"
                    exp["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    logging.warning(
                        f"Stale remote experiment '{exp['name']}' on {srv_name} "
                        f"(PID {exp['pid']} dead), marking failed"
                    )
                    exp["status"] = "failed"
                    exp["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    exp["error"] = f"Process died on {srv_name} (scheduler restart recovery)"


# ---------------------------------------------------------------------------
# Path remapping
# ---------------------------------------------------------------------------

def _remap_value(value, path_mappings: dict[str, str]):
    """Recursively remap string values in a JSON-like structure."""
    if isinstance(value, str):
        for local_prefix, remote_prefix in path_mappings.items():
            value = value.replace(local_prefix, remote_prefix)
        return value
    elif isinstance(value, dict):
        return {k: _remap_value(v, path_mappings) for k, v in value.items()}
    elif isinstance(value, list):
        return [_remap_value(item, path_mappings) for item in value]
    return value


def _remap_config(config_path: str, server: ServerConfig) -> str:
    """Read a config JSON, recursively remap all string path values for the remote server.
    Returns the remapped JSON string."""
    with open(config_path) as f:
        config = json.load(f)

    config = _remap_value(config, server.path_mappings)

    return json.dumps(config, indent=2)


# ---------------------------------------------------------------------------
# Git worktree management
# ---------------------------------------------------------------------------

def _sanitize_branch_name(branch: str) -> str:
    """Sanitize branch name for filesystem use: exp/sepsis-v2 → exp__sepsis-v2."""
    return branch.replace("/", "__").strip(".")


def _get_current_branch() -> str:
    """Get the current branch of the main repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(GIT_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _ensure_local_worktree(branch: str) -> Path | None:
    """Ensure a local git worktree exists for the given branch.
    Returns deep_pipeline path, or None on failure."""
    if branch == _get_current_branch():
        return REPO

    sanitized = _sanitize_branch_name(branch)
    wt_path = WORKTREE_BASE / sanitized
    dp_path = wt_path / "deep_pipeline"

    if (wt_path / ".git").exists():
        logging.info(f"Updating local worktree for '{branch}'")
        result = subprocess.run(
            ["git", "-C", str(wt_path), "reset", "--hard", branch],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logging.error(f"Failed to update local worktree for '{branch}': {result.stderr}")
            return None
    else:
        logging.info(f"Creating local worktree for '{branch}' at {wt_path}")
        WORKTREE_BASE.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "-C", str(GIT_ROOT), "worktree", "add", str(wt_path), branch],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logging.error(f"Failed to create local worktree for '{branch}': {result.stderr}")
            return None

    if not dp_path.exists():
        logging.error(f"Worktree created but deep_pipeline/ not found at {dp_path}")
        return None

    return dp_path


def _ensure_remote_worktree(branch: str, server: ServerConfig) -> str | None:
    """Ensure a remote git worktree exists for the given branch.
    Returns remote deep_pipeline path, or None on failure.
    Always creates a worktree — the local current branch does NOT imply
    the remote default path has the same code."""

    sanitized = _sanitize_branch_name(branch)
    remote_git_root = str(Path(server.repo_path).parent)
    remote_wt_base = str(Path(remote_git_root).parent / "EHR_Translator_worktrees")
    remote_wt = f"{remote_wt_base}/{sanitized}"
    remote_dp = f"{remote_wt}/deep_pipeline"

    logging.info(f"Fetching origin on {server.name} for branch '{branch}'")
    rc, _ = _ssh_run(server.host, f"git -C {remote_git_root} fetch origin", timeout=60)
    if rc != 0:
        logging.error(f"Failed to fetch origin on {server.name}")
        return None

    rc, out = _ssh_run(server.host, f"test -f {remote_wt}/.git && echo EXISTS")
    if rc == 0 and "EXISTS" in out:
        logging.info(f"Updating remote worktree for '{branch}' on {server.name}")
        rc, out = _ssh_run(server.host, f"git -C {remote_wt} reset --hard origin/{branch}", timeout=30)
        if rc != 0:
            logging.error(f"Failed to update remote worktree on {server.name}: {out}")
            return None
    else:
        logging.info(f"Creating remote worktree for '{branch}' on {server.name}")
        rc, out = _ssh_run(
            server.host,
            f"mkdir -p {remote_wt_base} && git -C {remote_git_root} worktree add {remote_wt} origin/{branch}",
            timeout=60,
        )
        if rc != 0:
            logging.error(f"Failed to create remote worktree on {server.name}: {out}")
            return None

    # Verify commit hash matches origin/<branch>
    rc_wt, wt_hash = _ssh_run(server.host, f"git -C {remote_wt} rev-parse HEAD", timeout=10)
    rc_orig, origin_hash = _ssh_run(
        server.host, f"git -C {remote_git_root} rev-parse origin/{branch}", timeout=10,
    )
    if rc_wt != 0 or rc_orig != 0:
        logging.warning(f"Could not verify commit hashes on {server.name} for '{branch}'")
    elif wt_hash.strip() != origin_hash.strip():
        logging.warning(
            f"Worktree hash mismatch on {server.name}: "
            f"worktree={wt_hash.strip()[:12]} vs origin={origin_hash.strip()[:12]}. "
            f"Force checking out origin/{branch}."
        )
        rc, out = _ssh_run(
            server.host,
            f"git -C {remote_wt} checkout -f origin/{branch}",
            timeout=30,
        )
        if rc != 0:
            logging.error(f"Force checkout failed on {server.name}: {out}")
            return None
        # Re-verify
        rc_wt2, wt_hash2 = _ssh_run(server.host, f"git -C {remote_wt} rev-parse HEAD", timeout=10)
        if rc_wt2 != 0 or wt_hash2.strip() != origin_hash.strip():
            logging.error(
                f"Commit hash still mismatched after force checkout on {server.name}. Aborting."
            )
            return None
        logging.info(f"Force checkout succeeded on {server.name}: {wt_hash2.strip()[:12]}")
    else:
        logging.info(f"Remote worktree on {server.name} at commit {wt_hash.strip()[:12]}")

    rc, _ = _ssh_run(server.host, f"test -d {remote_dp}")
    if rc != 0:
        logging.error(f"Remote worktree missing deep_pipeline/ at {remote_dp}")
        return None

    return remote_dp


def _prepare_worktree_config(exp: dict, worktree_dp: Path) -> str | None:
    """Read config from worktree, override run_dir to main tree, write to .worktree_configs/.
    Returns absolute path to prepared config, or None on failure."""
    config_path = worktree_dp / exp["config"]
    if not config_path.exists():
        logging.error(f"Config not found in worktree: {config_path}")
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.error(f"Failed to read config from worktree: {e}")
        return None

    output = config.setdefault("output", {})
    rel_run_dir = output.get("run_dir", f"runs/{exp['name']}")
    output["run_dir"] = str(REPO / rel_run_dir)
    if "log_file" in output:
        output["log_file"] = str(REPO / output["log_file"])

    WORKTREE_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    prepared_path = WORKTREE_CONFIGS_DIR / f"{exp['name']}.json"
    with open(prepared_path, "w") as f:
        json.dump(config, f, indent=2)

    return str(prepared_path)


def cleanup_worktrees(branch: str = "", servers: dict[str, ServerConfig] | None = None):
    """Remove worktrees. If branch is specified, only remove that branch's worktree."""
    if not WORKTREE_BASE.exists():
        print("No local worktrees found.")
        return

    if branch:
        sanitized = _sanitize_branch_name(branch)
        targets = [WORKTREE_BASE / sanitized]
        targets = [t for t in targets if t.exists()]
        if not targets:
            print(f"No worktree found for branch '{branch}'")
    else:
        targets = [p for p in WORKTREE_BASE.iterdir() if p.is_dir()]

    for wt in targets:
        logging.info(f"Removing local worktree: {wt}")
        result = subprocess.run(
            ["git", "-C", str(GIT_ROOT), "worktree", "remove", "--force", str(wt)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logging.warning(f"Failed to remove worktree {wt}: {result.stderr}")
        else:
            print(f"Removed local worktree: {wt}")

    if not servers:
        return
    for srv_name, server in servers.items():
        if server.is_local:
            continue
        remote_git_root = str(Path(server.repo_path).parent)
        remote_wt_base = str(Path(remote_git_root).parent / "EHR_Translator_worktrees")

        if branch:
            sanitized = _sanitize_branch_name(branch)
            remote_wt = f"{remote_wt_base}/{sanitized}"
            rc, _ = _ssh_run(server.host, f"test -d {remote_wt}")
            if rc == 0:
                logging.info(f"Removing remote worktree on {server.name}: {remote_wt}")
                rc, out = _ssh_run(
                    server.host,
                    f"git -C {remote_git_root} worktree remove --force {remote_wt}",
                    timeout=30,
                )
                if rc == 0:
                    print(f"Removed remote worktree on {srv_name}: {remote_wt}")
                else:
                    logging.warning(f"Failed to remove remote worktree on {srv_name}: {out}")
        else:
            rc, listing = _ssh_run(server.host, f"ls -d {remote_wt_base}/*/ 2>/dev/null || true")
            if rc == 0 and listing.strip():
                for remote_wt in listing.strip().split("\n"):
                    remote_wt = remote_wt.rstrip("/")
                    logging.info(f"Removing remote worktree on {server.name}: {remote_wt}")
                    _ssh_run(
                        server.host,
                        f"git -C {remote_git_root} worktree remove --force {remote_wt}",
                        timeout=30,
                    )


# ---------------------------------------------------------------------------
# Experiment launch & monitoring
# ---------------------------------------------------------------------------

def launch_experiment_local(exp: dict, gpu: int,
                            repo_path: Path | None = None,
                            config_override: str | None = None) -> subprocess.Popen:
    """Launch an experiment on a local GPU."""
    repo = repo_path or REPO
    task = infer_task(exp["config"])
    log_name = f"{exp['name']}_{task}.log"
    log_path = LOG_DIR / log_name

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EHR_LOG_FILE"] = str(log_path)

    config_path = config_override or str(repo / exp["config"])
    output_path = str(REPO / exp["output"])  # Always main tree

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    run_command = exp.get("command", "train_and_eval")
    cmd = [
        sys.executable, str(repo / "run.py"), run_command,
        "--config", config_path,
        "--output_parquet", output_path,
    ]

    logging.info(f"Launching '{exp['name']}' on local GPU {gpu}: {' '.join(cmd)}")
    logging.info(f"  Log: {log_path}")
    if repo_path and repo_path != REPO:
        logging.info(f"  Worktree: {repo}")

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
        cwd=str(repo),
    )

    exp["status"] = "running"
    exp["server"] = "local"
    exp["gpu"] = gpu
    exp["pid"] = proc.pid
    exp["started"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _running_procs[exp["name"]] = proc
    return proc


def _write_remote_file(server: ServerConfig, remote_path: str, content: str) -> bool:
    """Write content to a file on a remote server via SSH stdin pipe."""
    try:
        result = subprocess.run(
            ["ssh"] + _SSH_OPTS + [server.host,
             f"mkdir -p $(dirname {remote_path}) && cat > {remote_path}"],
            input=content, text=True, capture_output=True, timeout=15,
        )
        return result.returncode == 0
    except Exception as e:
        logging.warning(f"Failed to write {remote_path} on {server.name}: {e}")
        return False


def launch_experiment_remote(exp: dict, gpu: int, server: ServerConfig,
                             config_source: str | None = None,
                             remote_cwd: str | None = None) -> bool:
    """Launch an experiment on a remote server via SSH. Returns True on success."""
    task = infer_task(exp["config"])
    config_path = config_source or str(REPO / exp["config"])
    remote_code_dir = remote_cwd or server.repo_path
    remote_main = server.repo_path  # Always centralized for outputs

    # Remap config paths for remote
    remapped_json = _remap_config(config_path, server)

    # Remote paths — always centralized in main repo
    remote_config_dir = f"{remote_main}/experiments/.remote_configs"
    remote_config_path = f"{remote_config_dir}/{exp['name']}.json"
    remote_output = f"{remote_main}/{exp['output']}"
    remote_log = f"{remote_main}/experiments/logs/{exp['name']}_{task}.log"

    # Write remapped config to remote via stdin pipe (avoids heredoc quoting issues)
    if not _write_remote_file(server, remote_config_path, remapped_json):
        logging.error(f"Failed to write remote config for '{exp['name']}' on {server.name}")
        exp["status"] = "failed"
        exp["error"] = f"Failed to write remote config on {server.name}"
        return False

    # Create output and log directories
    _ssh_run(server.host, f"mkdir -p $(dirname {remote_output}) && mkdir -p $(dirname {remote_log})", timeout=10)

    # Launch via Popen (non-blocking SSH) + PID file
    # Conda-activated Python processes keep SSH FDs open, so we can't use
    # subprocess.run (it would block). Instead, fire-and-forget with Popen
    # and read the PID from a file written by the remote shell.
    run_command = exp.get("command", "train_and_eval")
    pid_file = f"{remote_main}/experiments/.remote_configs/{exp['name']}.pid"
    conda_activate = f"source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate {server.conda_env}"
    launch_cmd = (
        f"{conda_activate} && "
        f"cd {remote_code_dir} && "
        f"CUDA_VISIBLE_DEVICES={gpu} EHR_LOG_FILE={remote_log} "
        f"nohup python run.py {run_command} "
        f"--config {remote_config_path} "
        f"--output_parquet {remote_output} "
        f"> {remote_log} 2>&1 & "
        f"echo $! > {pid_file}"
    )

    # Fire-and-forget: Popen won't block waiting for the remote process
    ssh_proc = subprocess.Popen(
        ["ssh"] + _SSH_OPTS_NO_CONTROL + [server.host, launch_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )
    # Store for cleanup (will be reaped later)
    _running_procs[f"_ssh_{exp['name']}"] = ssh_proc

    # Wait for PID file to appear
    time.sleep(5)

    rc, pid_str = _ssh_run(server.host, f"cat {pid_file} 2>/dev/null", timeout=10)
    if rc != 0 or not pid_str.strip():
        logging.error(f"Failed to read PID for '{exp['name']}' on {server.name}")
        exp["status"] = "failed"
        exp["error"] = f"PID read failed on {server.name}"
        return False

    try:
        pid = int(pid_str.strip())
    except ValueError:
        logging.error(f"Invalid PID '{pid_str}' for '{exp['name']}' on {server.name}")
        exp["status"] = "failed"
        exp["error"] = f"PID parse failed on {server.name}"
        return False

    # Verify process is actually running
    if not pid_is_alive_remote(server, pid):
        logging.error(f"Remote process {pid} for '{exp['name']}' died immediately on {server.name}")
        exp["status"] = "failed"
        exp["error"] = f"Process died immediately on {server.name}"
        return False

    logging.info(f"Launched '{exp['name']}' on {server.name} GPU {gpu} (PID {pid})")
    logging.info(f"  Remote log: {remote_log}")
    if remote_cwd and remote_cwd != server.repo_path:
        logging.info(f"  Remote worktree: {remote_code_dir}")

    exp["status"] = "running"
    exp["server"] = server.name
    exp["gpu"] = gpu
    exp["pid"] = pid
    exp["started"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp["remote_config"] = remote_config_path

    return True


def launch_experiment(exp: dict, gpu: int, server: ServerConfig,
                      repo_path: Path | None = None,
                      config_override: str | None = None,
                      remote_cwd: str | None = None) -> bool:
    """Launch an experiment on the given server + GPU. Returns True on success."""
    if server.is_local:
        launch_experiment_local(exp, gpu, repo_path=repo_path, config_override=config_override)
        return True
    else:
        return launch_experiment_remote(exp, gpu, server,
                                        config_source=config_override,
                                        remote_cwd=remote_cwd)


def collect_results(exp: dict, servers: dict[str, ServerConfig]):
    """Collect results using collect_result.py after experiment finishes."""
    task = infer_task(exp["config"])
    if task == "unknown":
        logging.warning(f"Cannot collect results for '{exp['name']}': unknown task")
        return

    srv_name = exp.get("server", "local")
    server = servers.get(srv_name)

    # For remote experiments, rsync the log file first
    if server and not server.is_local:
        remote_log = f"{server.repo_path}/experiments/logs/{exp['name']}_{task}.log"
        local_log = LOG_DIR / f"{exp['name']}_{task}.log"
        try:
            subprocess.run(
                ["rsync", "-az", f"{server.host}:{remote_log}", str(local_log)],
                timeout=60, capture_output=True,
            )
        except Exception as e:
            logging.warning(f"Failed to rsync log for '{exp['name']}' from {srv_name}: {e}")

    try:
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "collect_result.py"),
             exp["name"], task],
            cwd=str(REPO),
            timeout=30,
            capture_output=True,
        )
    except Exception as e:
        logging.warning(f"collect_result.py failed for '{exp['name']}': {e}")
        return

    result_path = RESULTS_DIR / f"{exp['name']}_{task}.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            diff = data.get("difference", {})
            if diff:
                exp["results"] = diff
                logging.info(f"  Results for '{exp['name']}': {diff}")
        except json.JSONDecodeError:
            logging.warning(f"Malformed results JSON for '{exp['name']}'")


def check_running(experiments: list[dict], servers: dict[str, ServerConfig]):
    """Check running experiments and update status on completion."""
    # Batch remote PID checks by server
    remote_checks: dict[str, list[dict]] = {}

    for exp in experiments:
        if exp.get("status") != "running":
            continue

        name = exp["name"]
        srv_name = exp.get("server", "local")
        server = servers.get(srv_name)

        if server and server.is_local:
            # Local experiment — check via proc or PID
            proc = _running_procs.get(name)
            if proc is not None:
                retcode = proc.poll()
                if retcode is None:
                    continue  # Still running
                finished_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                exp["finished"] = finished_ts

                if retcode == 0:
                    logging.info(f"Experiment '{name}' completed successfully")
                    exp["status"] = "done"
                    collect_results(exp, servers)
                else:
                    logging.error(f"Experiment '{name}' failed (exit code {retcode})")
                    exp["status"] = "failed"
                    exp["error"] = f"Exit code {retcode}"

                _running_procs.pop(name, None)
            else:
                pid = exp.get("pid")
                if pid and not pid_is_alive(pid):
                    logging.warning(f"Experiment '{name}' PID {pid} no longer running")
                    exp["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Check if experiment actually completed (output file exists)
                    output_path = REPO / exp.get("output", "")
                    if output_path.exists():
                        logging.info(f"Experiment '{name}' output found — marking as done")
                        exp["status"] = "done"
                        collect_results(exp, servers)
                    else:
                        exp["status"] = "failed"
                        exp["error"] = "Process died (detected during monitoring)"
                    _running_procs.pop(name, None)
        elif server and not server.is_local:
            # Queue for batch remote check
            remote_checks.setdefault(srv_name, []).append(exp)

    # Batch check remote experiments
    for srv_name, exps in remote_checks.items():
        server = servers[srv_name]
        pids = [e["pid"] for e in exps if e.get("pid")]
        if not pids:
            continue

        alive_map = batch_pid_check_remote(server, pids)
        for exp in exps:
            pid = exp.get("pid")
            if pid and not alive_map.get(pid, True):
                # Process finished — check exit status via log
                name = exp["name"]
                logging.info(f"Remote experiment '{name}' on {srv_name} finished (PID {pid})")
                exp["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Try to determine success by checking remote log for completion marker
                task = infer_task(exp["config"])
                remote_log = f"{server.repo_path}/experiments/logs/{name}_{task}.log"
                rc, tail = _ssh_run(
                    server.host,
                    f"tail -20 {remote_log} 2>/dev/null",
                )
                tail_lower = tail.lower()
                success = any(m in tail_lower for m in [
                    "evaluation results", "exported", "evaluation complete", "saved to",
                ])
                # Also check if the output parquet exists on the remote server
                remote_output = f"{server.repo_path}/{exp.get('output', '')}"
                rc_out, _ = _ssh_run(server.host, f"test -f {remote_output}")
                output_exists = (rc_out == 0)

                if success or output_exists:
                    exp["status"] = "done"
                    logging.info(f"  Remote experiment '{name}' completed successfully"
                                 f" (log_markers={success}, output_exists={output_exists})")
                else:
                    exp["status"] = "failed"
                    exp["error"] = f"Process died on {srv_name} (no success markers in log, no output file)"
                    logging.warning(f"  Remote experiment '{name}' FAILED — no success markers and no output file")
                if exp["status"] == "done":
                    collect_results(exp, servers)


# ---------------------------------------------------------------------------
# Server selection
# ---------------------------------------------------------------------------

def _select_server(
    exp: dict,
    servers: dict[str, ServerConfig],
    server_slots: dict[str, list[int]],
) -> tuple[str, int] | None:
    """Find a server+GPU for an experiment. Returns (server_name, gpu) or None."""
    preferred = exp.get("server", "any")

    if preferred != "any":
        # Pinned to a specific server
        server = servers.get(preferred)
        if server and server.slurm:
            logging.warning(
                f"Experiment '{exp['name']}' pinned to SLURM server '{preferred}' — "
                f"use athena_submit.py instead. Skipping."
            )
            return None
        slots = server_slots.get(preferred, [])
        if slots:
            return preferred, slots.pop(0)
        return None

    # Auto-assign: iterate servers in definition order, skip SLURM servers
    for srv_name, slots in server_slots.items():
        server = servers.get(srv_name)
        if server and server.slurm:
            continue  # SLURM servers are never auto-assigned
        if slots:
            return srv_name, slots.pop(0)
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def scheduler_loop(dry_run: bool = False):
    """Main scheduler loop — poll GPUs, launch pending experiments."""
    global _shutdown

    with _queue_lock():
        queue = load_queue()
        settings = queue.get("settings", {})
        experiments = queue.get("experiments", [])
        servers = _parse_servers(settings)

        # Recover stale entries on startup
        recover_stale(experiments, servers)
        if not dry_run:
            save_queue(queue)

    logging.info(
        f"Scheduler started — {sum(1 for e in experiments if e.get('status') == 'pending')} pending, "
        f"{sum(1 for e in experiments if e.get('status') == 'running')} running, "
        f"{sum(1 for e in experiments if e.get('status') == 'done')} done"
    )
    server_list = ', '.join(f'{s.name} ({s.host or "local"})' for s in servers.values())
    logging.info(f"Servers: {server_list}")

    poll_interval = settings.get("poll_interval", 60)
    threshold_mb = settings.get("gpu_free_threshold_mb", 1000)

    while not _shutdown:
        with _queue_lock():
            # Reload queue (may have been edited externally)
            try:
                queue = load_queue()
                settings = queue.get("settings", {})
                experiments = queue.get("experiments", [])
                servers = _parse_servers(settings)
            except Exception as e:
                logging.error(f"Failed to reload queue: {e}")
                time.sleep(poll_interval)
                continue

            # Check running experiments
            check_running(experiments, servers)

            # Build per-server available GPU slots
            # First, find which GPUs each server is already using
            running_by_server: dict[str, set[int]] = {name: set() for name in servers}
            for exp in experiments:
                if exp.get("status") == "running" and "gpu" in exp:
                    srv = exp.get("server", "local")
                    if srv in running_by_server:
                        running_by_server[srv].add(exp["gpu"])

            # Get free GPUs and compute available slots per server
            server_slots: dict[str, list[int]] = {}
            for srv_name, server in servers.items():
                free_gpus = get_free_gpus(threshold_mb, server)
                available = select_gpus(server, free_gpus, running_by_server.get(srv_name, set()))
                server_slots[srv_name] = available

            # Find pending experiments
            pending = [e for e in experiments if e.get("status") == "pending"]

            if dry_run:
                _print_dry_run(servers, server_slots, running_by_server, pending, experiments, threshold_mb)
                return

            # Launch as many as we have GPU slots
            launched = 0
            for exp in list(pending):
                result = _select_server(exp, servers, server_slots)
                if result is None:
                    continue  # No slots available for this experiment

                srv_name, gpu = result
                server = servers[srv_name]

                # Resolve branch worktree
                branch = exp.get("branch")
                repo_path = REPO
                config_override = None
                remote_cwd = None
                needs_local_worktree = branch and branch != _get_current_branch()
                # Remote servers ALWAYS need a worktree when branch is specified,
                # because _get_current_branch() reflects the LOCAL checkout —
                # the remote default path may have different code.
                needs_remote_worktree = branch and not server.is_local

                if needs_local_worktree:
                    repo_path = _ensure_local_worktree(branch)
                    if repo_path is None:
                        exp["status"] = "failed"
                        exp["error"] = f"Local worktree failed for branch '{branch}'"
                        continue

                if needs_local_worktree or needs_remote_worktree:
                    config_override = _prepare_worktree_config(exp, repo_path)
                    if config_override is None:
                        exp["status"] = "failed"
                        exp["error"] = f"Config preparation failed for branch '{branch}'"
                        continue

                if needs_remote_worktree:
                    remote_cwd = _ensure_remote_worktree(branch, server)
                    if remote_cwd is None:
                        exp["status"] = "failed"
                        exp["error"] = f"Remote worktree failed for branch '{branch}' on {srv_name}"
                        continue

                # Verify config exists
                check_path = config_override or str(repo_path / exp["config"])
                if not Path(check_path).exists():
                    logging.error(
                        f"Config not found for '{exp['name']}': {check_path}. Skipping."
                    )
                    exp["status"] = "failed"
                    exp["error"] = f"Config not found: {check_path}"
                    continue

                if launch_experiment(exp, gpu, server, repo_path=repo_path,
                                     config_override=config_override, remote_cwd=remote_cwd):
                    launched += 1
                pending.remove(exp)

            # Save updated queue
            save_queue(queue)

        if launched:
            logging.info(f"Launched {launched} experiment(s) this cycle")

        # Check if all done
        remaining = sum(1 for e in experiments
                        if e.get("status") in ("pending", "running"))
        if remaining == 0:
            logging.info("All experiments completed. Scheduler exiting.")
            return

        time.sleep(poll_interval)

    logging.info("Scheduler shut down by signal")


def _print_dry_run(servers, server_slots, running_by_server, pending, experiments, threshold_mb):
    """Print what the scheduler would do without launching anything."""
    hour = datetime.now().hour
    period = "daytime" if 9 <= hour < 21 else "nighttime"

    print(f"\n{'='*70}")
    print(f"  GPU Scheduler — Dry Run  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*70}")

    for srv_name, server in servers.items():
        max_gpus = get_max_gpus(server)
        running = running_by_server.get(srv_name, set())
        available = server_slots.get(srv_name, [])
        host_str = server.host or "localhost"
        print(f"\n  [{srv_name}] ({host_str})")
        print(f"    Period:      {period} (max {max_gpus} GPUs)")
        print(f"    Running:     {running or '{none}'}")
        print(f"    Available:   {available}")

    print(f"\n  Pending: {len(pending)} experiment(s)")

    # Simulate assignment
    slots_copy = {k: list(v) for k, v in server_slots.items()}
    if pending:
        print("\n  Would launch:")
        for exp in pending:
            result = _select_server(exp, servers, slots_copy)
            if result:
                srv_name, gpu = result
                pinned = exp.get("server", "any")
                pin_str = f" (pinned)" if pinned != "any" else ""
                print(f"    {srv_name} GPU {gpu} <- {exp['name']}{pin_str}")
            else:
                print(f"    [no slot]  <- {exp['name']}")

    # Summary table
    print(f"\n  {'Status':<10} {'Count'}")
    print(f"  {'-'*20}")
    for status in ["pending", "running", "done", "failed"]:
        count = sum(1 for e in experiments if e.get("status") == status)
        if count:
            print(f"  {status:<10} {count}")
    print()


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status():
    """Print a human-readable status table."""
    queue = load_queue()
    experiments = queue.get("experiments", [])
    settings = queue.get("settings", {})
    servers = _parse_servers(settings)

    hour = datetime.now().hour
    period = "daytime" if 9 <= hour < 21 else "nighttime"
    threshold_mb = settings.get("gpu_free_threshold_mb", 1000)

    print(f"\n{'='*70}")
    print(f"  Experiment Queue Status  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*70}")

    # Per-server GPU info
    for srv_name, server in servers.items():
        if server.slurm:
            host_str = server.host or "localhost"
            print(f"  [{srv_name}] {host_str}  |  SLURM-managed (use athena_submit.py)")
            # Query SLURM jobs
            if server.host:
                rc, squeue_out = _ssh_run(server.host, "squeue -u $USER -o '%.8i %.30j %.8T %.10M %.6D %R' 2>/dev/null", timeout=15)
                if rc == 0 and squeue_out.strip():
                    for line in squeue_out.strip().split("\n"):
                        print(f"    {line.strip()}")
                else:
                    print(f"    (no active SLURM jobs or SSH failed)")
            continue
        max_gpus = get_max_gpus(server)
        free_gpus = get_free_gpus(threshold_mb, server)
        host_str = server.host or "localhost"
        print(f"  [{srv_name}] {host_str}  |  {period} (max {max_gpus})  |  Free GPUs: {free_gpus}")

    if not experiments:
        print("\n  No experiments in queue.")
        print()
        return

    # Group by status
    for status in ["running", "pending", "done", "failed"]:
        group = [e for e in experiments if e.get("status") == status]
        if not group:
            continue

        status_label = {
            "running": "RUNNING",
            "pending": "PENDING",
            "done":    "DONE",
            "failed":  "FAILED",
        }[status]
        print(f"\n  [{status_label}]")

        for exp in group:
            name = exp["name"]

            if status == "running":
                gpu = exp.get("gpu", "?")
                pid = exp.get("pid", "?")
                started = exp.get("started", "?")
                branch = exp.get("branch", "")
                branch_str = f"  [{branch}]" if branch else ""
                srv_name = exp.get("server", "local")
                server = servers.get(srv_name)
                if server and not server.is_local:
                    alive = "alive" if pid_is_alive_remote(server, pid) else "DEAD"
                else:
                    alive = "alive" if exp.get("pid") and pid_is_alive(exp["pid"]) else "DEAD"
                print(f"    {name:<35} {srv_name} GPU {gpu}  PID {pid} ({alive}){branch_str}  started {started}")

            elif status == "pending":
                notes = exp.get("notes", "")
                branch = exp.get("branch", "")
                branch_str = f"[{branch}] " if branch else ""
                srv_pref = exp.get("server", "any")
                srv_str = f"[{srv_pref}] " if srv_pref != "any" else ""
                print(f"    {name:<35} {srv_str}{branch_str}{notes}")

            elif status == "done":
                finished = exp.get("finished", "?")
                results = exp.get("results", {})
                srv_name = exp.get("server", "")
                srv_str = f"({srv_name}) " if srv_name and srv_name != "local" else ""
                result_str = "  ".join(f"{k}: {v:+.4f}" for k, v in results.items()) if results else ""
                print(f"    {name:<35} {srv_str}finished {finished}  {result_str}")

            elif status == "failed":
                error = exp.get("error", "unknown")
                srv_name = exp.get("server", "")
                srv_str = f"({srv_name}) " if srv_name and srv_name != "local" else ""
                print(f"    {name:<35} {srv_str}error: {error}")

    print()


# ---------------------------------------------------------------------------
# Add experiment
# ---------------------------------------------------------------------------

def add_experiment(name: str, config: str, notes: str = "", server: str = "",
                   branch: str = ""):
    """Add a new pending experiment to the queue."""
    with _queue_lock():
        queue = load_queue()
        experiments = queue.get("experiments", [])

        # Check for duplicate name
        existing_names = {e["name"] for e in experiments}
        if name in existing_names:
            logging.error(f"Experiment '{name}' already exists in queue")
            sys.exit(1)

        # Default branch to current
        if not branch:
            branch = _get_current_branch()

        # Verify config exists (check in worktree if different branch)
        current = _get_current_branch()
        if branch and branch != current:
            wt_dp = _ensure_local_worktree(branch)
            config_abs = wt_dp / config if wt_dp else REPO / config
        else:
            config_abs = REPO / config
        if not config_abs.exists():
            logging.warning(f"Config file not found: {config_abs}")

        task = infer_task(config)
        output = f"experiments/results/{name}.parquet"

        entry = {
            "name": name,
            "config": config,
            "output": output,
            "status": "pending",
        }
        if branch:
            entry["branch"] = branch
        if server:
            entry["server"] = server
        if notes:
            entry["notes"] = notes

        # Insert before first non-pending entry (or at end)
        insert_idx = len(experiments)
        for i, exp in enumerate(experiments):
            if exp.get("status") not in ("pending",):
                insert_idx = i
                break
        experiments.insert(insert_idx, entry)

        save_queue(queue)
    srv_str = f", server={server}" if server else ""
    branch_str = f", branch={branch}" if branch else ""
    logging.info(f"Added experiment '{name}' ({task}{srv_str}{branch_str}) at position {insert_idx}")
    print(f"Added '{name}' to queue (position {insert_idx}, task={task}{srv_str}{branch_str})")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(signum, frame):
    global _shutdown
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    _shutdown = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU Experiment Scheduler (multi-server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--status", action="store_true",
                        help="Show queue status and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would launch without running")
    parser.add_argument("--add", action="store_true",
                        help="Add experiment to queue")
    parser.add_argument("--name", type=str, help="Experiment name (for --add)")
    parser.add_argument("--config", type=str, help="Config path (for --add)")
    parser.add_argument("--notes", type=str, default="", help="Notes (for --add)")
    parser.add_argument("--server", type=str, default="",
                        help="Pin to server (for --add). Default: any")
    parser.add_argument("--branch", type=str, default="",
                        help="Git branch for experiment (for --add). Default: current branch")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove worktrees (optionally filter by --branch)")

    args = parser.parse_args()

    setup_logging()

    if args.cleanup:
        queue = load_queue()
        servers = _parse_servers(queue.get("settings", {}))
        cleanup_worktrees(args.branch, servers)
        return

    if args.status:
        show_status()
        return

    if args.add:
        if not args.name or not args.config:
            parser.error("--add requires --name and --config")
        add_experiment(args.name, args.config, args.notes, args.server, args.branch)
        return

    if args.dry_run:
        scheduler_loop(dry_run=True)
        return

    # Daemon mode
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logging.info("Starting GPU scheduler daemon (Ctrl+C to stop)")
    try:
        scheduler_loop()
    except KeyboardInterrupt:
        logging.info("Interrupted, shutting down...")
    finally:
        # Wait briefly for any running processes (don't kill them)
        if _running_procs:
            logging.info(
                f"{len(_running_procs)} experiment(s) still running — "
                "they will continue in background"
            )


if __name__ == "__main__":
    main()
