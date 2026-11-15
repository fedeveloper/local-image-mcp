"""Fake backend: writes a solid-colour PNG without a GPU. Used by tests and CI."""

from __future__ import annotations

import asyncio
import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any

from local_image_mcp import __version__
from local_image_mcp.backends.base import BackendHealth, ProgressCallback
from local_image_mcp.errors import JobCancelledError
from local_image_mcp.resolver import ResolvedModel
from local_image_mcp.types import BackendOutput, ProgressEvent, ProgressStage, ResolvedSettings


class FakeBackend:
    """Deterministic: the same prompt and seed always give the same colour.

    Option ``step_delay`` (seconds, default 0) slows each step, to test progress and cancel.
    """

    name = "fake"

    def check(self) -> BackendHealth:
        return BackendHealth(name=self.name, available=True, version=__version__)

    async def generate(
        self,
        settings: ResolvedSettings,
        model: ResolvedModel,
        output_file: Path,
        options: dict[str, Any],
        on_progress: ProgressCallback,
        cancel: asyncio.Event,
    ) -> BackendOutput:
        delay = float(options.get("step_delay", 0.0))
        on_progress(ProgressEvent(stage=ProgressStage.LOADING))
        for step in range(1, settings.steps + 1):
            if cancel.is_set():
                raise JobCancelledError("Job cancelled.")
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
            on_progress(
                ProgressEvent(stage=ProgressStage.SAMPLING, step=step, total_steps=settings.steps)
            )
        if cancel.is_set():
            raise JobCancelledError("Job cancelled.")
        on_progress(ProgressEvent(stage=ProgressStage.DECODING))
        colour = solid_colour(settings.prompt, settings.seed)
        png = solid_png(settings.width, settings.height, colour)
        await asyncio.to_thread(output_file.write_bytes, png)
        return BackendOutput(
            image_path=output_file,
            backend_name=self.name,
            backend_version=__version__,
            options={"step_delay": delay},
            native_record={"colour": list(colour)},
        )


def solid_colour(prompt: str, seed: int) -> tuple[int, int, int]:
    digest = hashlib.sha256(f"{seed}:{prompt}".encode()).digest()
    return digest[0], digest[1], digest[2]


def solid_png(width: int, height: int, colour: tuple[int, int, int]) -> bytes:
    """Encode an RGB PNG of one colour with the standard library only."""
    row = b"\x00" + bytes(colour) * width
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
