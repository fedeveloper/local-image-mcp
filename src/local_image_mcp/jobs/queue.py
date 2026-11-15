"""Single-worker asynchronous job queue.

``submit`` validates the request first, so that the caller gets errors at once.
One worker runs one job at a time; ``GpuLock`` extends that rule across processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from local_image_mcp.errors import JobCancelledError, JobNotFoundError, LocalImageError
from local_image_mcp.jobs.gpu_lock import GpuLock
from local_image_mcp.jobs.model import Job, JobState, utc_now
from local_image_mcp.jobs.store import JobStore
from local_image_mcp.registry import FileRef
from local_image_mcp.service import ImageService, PreparedJob
from local_image_mcp.types import GenerationRequest, ProgressEvent, ProgressStage

log = logging.getLogger(__name__)


@dataclass
class _Entry:
    job: Job
    prepared: PreparedJob
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)


class JobQueue:
    def __init__(self, service: ImageService, store: JobStore, gpu_lock: GpuLock) -> None:
        self.service = service
        self.store = store
        self.gpu_lock = gpu_lock
        self._entries: dict[str, _Entry] = {}
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._order: list[str] = []
        self._worker: asyncio.Task[None] | None = None

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._worker is not None:
            return
        self.store.claim_ownership()
        self.store.recover(owned_ids=set(self._entries))
        self.store.prune()
        self._worker = asyncio.create_task(self._run_worker(), name="local-image-worker")

    async def stop(self) -> None:
        """Cancel every unfinished job and stop the worker."""
        for entry in self._entries.values():
            if not entry.job.state.is_terminal:
                entry.cancel.set()
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        for entry in self._entries.values():
            if not entry.job.state.is_terminal:
                self._finish(entry, JobState.CANCELLED, "The server stopped.")
        self.store.release_ownership()

    async def __aenter__(self) -> JobQueue:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ---------------------------------------------------------------- public

    def submit(
        self, request: GenerationRequest, file_overrides: dict[str, FileRef] | None = None
    ) -> Job:
        """Validate and queue a request. Raises ``LocalImageError`` for a bad request."""
        prepared = self.service.prepare(request, file_overrides)
        job = Job(request=request, warnings=list(prepared.warnings))
        job.progress = ProgressEvent(stage=ProgressStage.QUEUED)
        entry = _Entry(job=job, prepared=prepared)
        self._entries[job.id] = entry
        self._order.append(job.id)
        self.store.save(job)
        self._pending.put_nowait(job.id)
        return job

    def get(self, job_id: str) -> Job:
        entry = self._entries.get(job_id)
        if entry is not None:
            return entry.job
        job = self.store.load(job_id)
        if job is None:
            raise JobNotFoundError(f"No job with id '{job_id}'.")
        return job

    def queue_position(self, job_id: str) -> int | None:
        """1 = next to run. ``None`` when the job is not waiting in the queue."""
        waiting = [jid for jid in self._order if self._entries[jid].job.state is JobState.QUEUED]
        return waiting.index(job_id) + 1 if job_id in waiting else None

    async def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Wait until the job is finished, or until ``timeout`` seconds pass."""
        entry = self._entries.get(job_id)
        if entry is None:
            return self.get(job_id)
        if timeout is None:
            await entry.done.wait()
        elif timeout > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(entry.done.wait(), timeout)
        return entry.job

    async def cancel(self, job_id: str, timeout: float = 10.0) -> Job:
        entry = self._entries.get(job_id)
        if entry is None:
            job = self.get(job_id)  # raises if unknown
            return job  # a job from a previous run is already terminal
        if entry.job.state.is_terminal:
            return entry.job
        entry.cancel.set()
        if entry.job.state is JobState.QUEUED:
            self._finish(entry, JobState.CANCELLED, "Cancelled before it started.")
            return entry.job
        return await self.wait(job_id, timeout)

    # ---------------------------------------------------------------- worker

    async def _run_worker(self) -> None:
        while True:
            job_id = await self._pending.get()
            entry = self._entries.get(job_id)
            if entry is None or entry.job.state is not JobState.QUEUED:
                continue
            await self._run_one(entry)

    async def _run_one(self, entry: _Entry) -> None:
        job = entry.job
        job.state = JobState.RUNNING
        job.started_at = utc_now()
        self._progress(entry, ProgressEvent(stage=ProgressStage.WAITING_FOR_GPU))
        try:
            await self.gpu_lock.acquire(entry.cancel)
        except JobCancelledError as exc:
            self._finish(entry, JobState.CANCELLED, exc.message)
            return
        try:
            result = await self.service.run(
                entry.prepared, lambda ev: self._progress(entry, ev), entry.cancel
            )
        except JobCancelledError as exc:
            self._finish(entry, JobState.CANCELLED, exc.message)
        except LocalImageError as exc:
            self._finish(entry, JobState.FAILED, exc.message)
        except asyncio.CancelledError:
            self._finish(entry, JobState.CANCELLED, "The server stopped.")
            raise
        except Exception as exc:
            log.exception("job %s failed", job.id)
            self._finish(entry, JobState.FAILED, f"Unexpected error: {exc!r}")
        else:
            job.result = result
            self._finish(entry, JobState.SUCCEEDED, None)
        finally:
            self.gpu_lock.release()

    def _progress(self, entry: _Entry, event: ProgressEvent) -> None:
        entry.job.progress = event
        self.store.save(entry.job)

    def _finish(self, entry: _Entry, state: JobState, error: str | None) -> None:
        job = entry.job
        job.state = state
        job.error = error
        job.finished_at = utc_now()
        if state is JobState.SUCCEEDED:
            job.progress = ProgressEvent(stage=ProgressStage.DONE)
        self.store.save(job)
        entry.done.set()
