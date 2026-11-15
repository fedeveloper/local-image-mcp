"""The backend interface. A backend turns resolved settings and model files into an image."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from local_image_mcp.resolver import ResolvedModel
from local_image_mcp.types import BackendOutput, ProgressEvent, ResolvedSettings

ProgressCallback = Callable[[ProgressEvent], None]


class BackendHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    available: bool
    version: str | None = None
    detail: str | None = None


@runtime_checkable
class Backend(Protocol):
    """Contract for all backends.

    ``generate`` must:

    - write a PNG to exactly ``output_file``;
    - call ``on_progress`` when the stage or step changes (best effort);
    - stop soon after ``cancel`` is set, free GPU memory, and raise ``JobCancelledError``.
    """

    name: str

    def check(self) -> BackendHealth: ...

    async def generate(
        self,
        settings: ResolvedSettings,
        model: ResolvedModel,
        output_file: Path,
        options: dict[str, Any],
        on_progress: ProgressCallback,
        cancel: asyncio.Event,
    ) -> BackendOutput: ...
