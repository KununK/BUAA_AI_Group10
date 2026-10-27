"""Launch a recorded local process and retain its real exit code."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.common import baseline as b


def identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if text[0] == "Z" else text[19]
    except (OSError, IndexError):
        return None


def evaluation_lock_path(command, gpu=None, directory=None):
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES') if gpu is None else gpu
    if gpu not in ('0', '1') or '-m' not in command:
        return None
    index = command.index('-m') + 1
    if index >= len(command) or not command[index].startswith('scripts.eval.'):
        return None
    return Path(directory or b.ROOT / 'outputs') / f'.gpu_eval_{gpu}.lock'


@contextmanager
def evaluation_lock(command, gpu=None, directory=None):
    """Serialize single-GPU evaluators, including their CUDA process shutdown."""
    path = evaluation_lock_path(command, gpu, directory)
    if path is None:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Specify a command after --")
    out = args.job_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.supervise:
        path = evaluation_lock_path(command)
        if path is not None:
            b.write_json(out / 'gpu_eval_lock.json', {'status': 'waiting', 'path': str(path)})
        with evaluation_lock(command):
            if path is not None:
                b.write_json(out / 'gpu_eval_lock.json', {'status': 'acquired', 'path': str(path)})
            child = subprocess.Popen(command, cwd=b.ROOT)
            b.write_json(out / "worker.json", {
                "pid": child.pid, "kernel_start_time": identity(child.pid),
                "command": command, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            })
            code = child.wait()
        if path is not None:
            b.write_json(out / 'gpu_eval_lock.json', {'status': 'released', 'path': str(path)})
        b.write_json(out / "exit.json", {
            "exit_code": code, "worker_pid": child.pid,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise SystemExit(code)
    with (out / "job.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This job is already running")
        if (out / "launch.json").exists():
            raise SystemExit("Use a new job directory; preserve the previous execution record")
        argv = [sys.executable, "-u", "-m", "scripts.common.jobs", "--supervise",
                "--job-dir", str(out), "--", *command]
        with (out / "console.log").open("a") as log:
            child = subprocess.Popen(
                argv, cwd=b.ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),),
            )
        record = {
            "supervisor_pid": child.pid, "kernel_start_time": identity(child.pid),
            "command": command, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started_utc": datetime.now(timezone.utc).isoformat(), "status": "launched_not_completed",
        }
        b.write_json(out / "launch.json", record)
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
