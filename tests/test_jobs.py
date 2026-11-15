from __future__ import annotations

import asyncio
import itertools
from pathlib import Path

import pytest
from filelock import FileLock

from local_image_mcp.errors import JobNotFoundError
from local_image_mcp.jobs import GpuLock, Job, JobQueue, JobState, JobStore
from local_image_mcp.jobs.store import INTERRUPTED_MESSAGE
from local_image_mcp.service import ImageService
from local_image_mcp.types import GenerationRequest, ProgressStage


def _queue(service: ImageService) -> JobQueue:
    s = service.settings
    return JobQueue(service, JobStore(s.jobs_dir), GpuLock(s.gpu_lock_path, poll_interval=0.01))


def _slow_fake(service: ImageService, delay: float = 0.05) -> None:
    spec = service.registry.models["fake"]
    service.registry.models["fake"] = spec.model_copy(
        update={"backend_options": {"step_delay": delay}}
    )


REQ = GenerationRequest(prompt="a lighthouse at dusk", model="fake", seed=9)


def test_job_runs_to_success_and_is_persisted(service: ImageService) -> None:
    async def scenario() -> Job:
        async with _queue(service) as queue:
            job = queue.submit(REQ)
            assert job.state is JobState.QUEUED
            return await queue.wait(job.id)

    job = asyncio.run(scenario())
    assert job.state is JobState.SUCCEEDED
    assert job.result is not None and job.result.image_path.is_file()
    assert job.percent == 100.0

    stored = JobStore(service.settings.jobs_dir).load(job.id)
    assert stored is not None and stored.state is JobState.SUCCEEDED
    # A new queue (for example after a restart) still finds the finished job.
    assert _queue(service).get(job.id).result == job.result


def test_validation_errors_happen_at_submit(service: ImageService) -> None:
    with pytest.raises(Exception, match="multiple of 16"):
        _queue(service).submit(REQ.model_copy(update={"width": 100}))


def test_one_job_at_a_time_and_queue_positions(service: ImageService) -> None:
    _slow_fake(service, 0.02)

    async def scenario() -> list[Job]:
        async with _queue(service) as queue:
            jobs = [queue.submit(REQ) for _ in range(3)]
            await asyncio.sleep(0.01)
            assert queue.queue_position(jobs[0].id) is None  # running
            assert queue.queue_position(jobs[1].id) == 1
            assert queue.queue_position(jobs[2].id) == 2
            return [await queue.wait(j.id) for j in jobs]

    jobs = asyncio.run(scenario())
    assert all(j.state is JobState.SUCCEEDED for j in jobs)
    for earlier, later in itertools.pairwise(jobs):
        assert earlier.finished_at is not None and later.started_at is not None
        assert earlier.finished_at <= later.started_at
    assert len({j.result.image_path for j in jobs if j.result}) == 3


def test_cancel_queued_and_running_jobs(service: ImageService) -> None:
    _slow_fake(service, 0.05)

    async def scenario() -> tuple[Job, Job]:
        async with _queue(service) as queue:
            running = queue.submit(REQ)
            queued = queue.submit(REQ)
            await asyncio.sleep(0.08)
            cancelled_queued = await queue.cancel(queued.id)
            cancelled_running = await queue.cancel(running.id)
            return cancelled_queued, cancelled_running

    queued, running = asyncio.run(scenario())
    assert queued.state is JobState.CANCELLED
    assert running.state is JobState.CANCELLED
    assert running.result is None


def test_wait_with_timeout_returns_running_job(service: ImageService) -> None:
    _slow_fake(service, 0.05)

    async def scenario() -> Job:
        async with _queue(service) as queue:
            job = queue.submit(REQ)
            waited = await queue.wait(job.id, timeout=0.12)
            return waited.model_copy(deep=True)  # snapshot before stop() cancels it

    job = asyncio.run(scenario())
    assert job.state is JobState.RUNNING
    assert job.progress is not None and job.progress.stage is ProgressStage.SAMPLING
    assert job.percent is not None and 0 < job.percent < 100


def test_gpu_lock_held_elsewhere_makes_job_wait(service: ImageService) -> None:
    other_process = FileLock(str(service.settings.gpu_lock_path), timeout=0)
    service.settings.gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
    other_process.acquire()
    try:

        async def scenario() -> Job:
            async with _queue(service) as queue:
                job = queue.submit(REQ)
                waiting = await queue.wait(job.id, timeout=0.1)
                assert waiting.progress is not None
                assert waiting.progress.stage is ProgressStage.WAITING_FOR_GPU
                return await queue.cancel(job.id)

        job = asyncio.run(scenario())
    finally:
        other_process.release()
    assert job.state is JobState.CANCELLED
    assert job.error is not None and "waited for the GPU" in job.error


def test_recover_marks_jobs_of_dead_owners_only(service: ImageService, tmp_path: Path) -> None:
    store = JobStore(service.settings.jobs_dir)
    dead = Job(request=REQ, state=JobState.RUNNING, owner_pid=999_999_001)
    alive = Job(request=REQ, state=JobState.QUEUED, owner_pid=999_999_002)
    store.save(dead)
    store.save(alive)
    alive_lock = FileLock(str(store.owners_dir / "999999002.lock"), timeout=0)
    alive_lock.acquire()
    try:
        changed = store.recover()
    finally:
        alive_lock.release()
    assert [j.id for j in changed] == [dead.id]
    reloaded = store.load(dead.id)
    assert reloaded is not None and reloaded.error == INTERRUPTED_MESSAGE
    still = store.load(alive.id)
    assert still is not None and still.state is JobState.QUEUED


def test_prune_keeps_recent_finished_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", keep_finished=2)
    for _ in range(4):
        store.save(Job(request=REQ, state=JobState.SUCCEEDED))
    store.save(Job(request=REQ, state=JobState.QUEUED, owner_pid=1))
    assert store.prune() == 2
    assert len(store.all()) == 3


def test_unknown_job(service: ImageService) -> None:
    with pytest.raises(JobNotFoundError):
        _queue(service).get("doesnotexist")
    with pytest.raises(JobNotFoundError):
        _queue(service).get("../../etc/passwd")
