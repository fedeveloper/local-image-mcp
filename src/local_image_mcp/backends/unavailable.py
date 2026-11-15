"""Placeholder for backends that are planned but not implemented yet."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from local_image_mcp.backends.base import BackendHealth, ProgressCallback
from local_image_mcp.errors import BackendNotAvailableError
from local_image_mcp.resolver import ResolvedModel
from local_image_mcp.types import BackendOutput, ResolvedSettings


class UnavailableBackend:
    """A backend that reports itself as not available and refuses to generate."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def check(self) -> BackendHealth:
        return BackendHealth(name=self.name, available=False, detail=self.reason)

    async def generate(
        self,
        settings: ResolvedSettings,
        model: ResolvedModel,
        output_file: Path,
        options: dict[str, Any],
        on_progress: ProgressCallback,
        cancel: asyncio.Event,
    ) -> BackendOutput:
        raise BackendNotAvailableError(f"Backend '{self.name}' is not available: {self.reason}")
