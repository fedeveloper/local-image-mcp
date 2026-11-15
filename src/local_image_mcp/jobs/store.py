"""Persist jobs as JSON files, so that results survive a server restart.

Several server processes (one per Claude Code session) can share one state directory.
Each live process holds a lock file ``owners/<pid>.lock``. A process only marks an
unfinished job as failed when the owner of that job no longer holds its lock.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

from filelock import FileLock, Timeout
from pydantic import ValidationError

from local_image_mcp.jobs.model import Job, JobState, utc_now

INTERRUPTED_MESSAGE = "The server stopped before the job finished."
_REPLACE_RETRIES = 5


class JobStore:
    def __init__(self, directory: Path, keep_finished: int = 500) -> None:
        self.directory = directory
        self.keep_finished = keep_finished
        self.owners_dir = directory / "owners"
        self.owners_dir.mkdir(parents=True, exist_ok=True)
        self._owner_lock: FileLock | None = None

    # ------------------------------------------------------------ ownership

    def claim_ownership(self) -> None:
        """Hold the owner lock of this process until ``release_ownership``."""
        if self._owner_lock is None:
            lock = FileLock(str(self._owner_path(os.getpid())), timeout=0)
            lock.acquire()
            self._owner_lock = lock

    def release_ownership(self) -> None:
        if self._owner_lock is not None:
            self._owner_lock.release()
            self._owner_lock = None

    def _owner_path(self, pid: int) -> Path:
        return self.owners_dir / f"{pid}.lock"

    def _owner_is_alive(self, pid: int) -> bool:
        if pid == os.getpid():
            return self._owner_lock is not None
        probe = FileLock(str(self._owner_path(pid)), timeout=0)
        try:
            probe.acquire()
        except Timeout:
            return True
        probe.release()
        return False

    # -------------------------------------------------------------- records

    def _path(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise ValueError(f"invalid job id: {job_id!r}")
        return self.directory / f"{job_id}.json"

    def save(self, job: Job) -> None:
        path = self._path(job.id)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        # On Windows, replace() fails while another process reads the file: retry briefly.
        for attempt in range(_REPLACE_RETRIES):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == _REPLACE_RETRIES - 1:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def load(self, job_id: str) -> Job | None:
        try:
            path = self._path(job_id)
        except ValueError:
            return None
        try:
            return Job.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None

    def all(self) -> list[Job]:
        jobs: list[Job] = []
        for path in self.directory.glob("*.json"):
            job = self.load(path.stem)
            if job is not None:
                jobs.append(job)
        return sorted(jobs, key=lambda j: j.created_at)

    def recover(self, owned_ids: set[str] | None = None) -> list[Job]:
        """Mark unfinished jobs of dead processes as failed. Returns the changed jobs.

        ``owned_ids`` lists jobs of a live queue in this process; they are never touched.
        """
        owned = owned_ids or set()
        changed: list[Job] = []
        for job in self.all():
            if job.state.is_terminal or job.id in owned:
                continue
            if job.owner_pid != os.getpid() and self._owner_is_alive(job.owner_pid):
                continue
            job.state = JobState.FAILED
            job.error = INTERRUPTED_MESSAGE
            job.finished_at = utc_now()
            self.save(job)
            changed.append(job)
        return changed

    def prune(self) -> int:
        """Delete the oldest finished job records above ``keep_finished``."""
        finished = [j for j in self.all() if j.state.is_terminal]
        excess = len(finished) - self.keep_finished
        removed = 0
        for job in finished[: max(excess, 0)]:
            with contextlib.suppress(OSError):
                self._path(job.id).unlink()
                removed += 1
        return removed
