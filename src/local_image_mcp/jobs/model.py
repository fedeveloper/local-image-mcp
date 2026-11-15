"""Job data model."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from local_image_mcp.types import GenerationRequest, GenerationResult, ProgressEvent


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


def new_job_id() -> str:
    return uuid.uuid4().hex[:16]


def utc_now() -> datetime:
    return datetime.now(UTC)


class Job(BaseModel):
    id: str = Field(default_factory=new_job_id)
    state: JobState = JobState.QUEUED
    request: GenerationRequest
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: ProgressEvent | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list[str])
    result: GenerationResult | None = None
    owner_pid: int = Field(default_factory=os.getpid)

    @property
    def percent(self) -> float | None:
        """Completion in percent, from the sampling steps; ``None`` if unknown."""
        if self.state is JobState.SUCCEEDED:
            return 100.0
        p = self.progress
        if p is None or p.step is None or not p.total_steps:
            return None
        return round(100.0 * p.step / p.total_steps, 1)
