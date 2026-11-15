"""A cross-process lock, so that only one job uses the GPU on this machine.

Each Claude Code session starts its own MCP server process. Without this lock, two
sessions could run two generations at the same time and exhaust 8 GB of VRAM.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from filelock import FileLock, Timeout

from local_image_mcp.errors import JobCancelledError


class GpuLock:
    def __init__(self, path: Path, poll_interval: float = 0.25) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(path), timeout=0)
        self.poll_interval = poll_interval

    @property
    def is_locked(self) -> bool:
        return self._lock.is_locked

    def try_acquire(self) -> bool:
        try:
            self._lock.acquire(timeout=0)
        except Timeout:
            return False
        return True

    async def acquire(self, cancel: asyncio.Event | None = None) -> None:
        """Wait for the lock. Raise ``JobCancelledError`` if ``cancel`` is set first."""
        while not self.try_acquire():
            if cancel is not None and cancel.is_set():
                raise JobCancelledError("Job cancelled while it waited for the GPU.")
            await asyncio.sleep(self.poll_interval)

    def release(self) -> None:
        self._lock.release()
