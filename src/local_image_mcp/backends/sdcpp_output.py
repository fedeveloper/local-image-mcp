"""Parse the console output of stable-diffusion.cpp ``sd-cli`` into progress events.

Verified against stable-diffusion.cpp commit 532d2c1 (2026-11-15), source files
``src/core/util.cpp`` (progress bar) and ``examples/common/log.cpp`` (log lines):

- Log lines: ``[I] message --- file.cpp:123``. ``[E]`` lines go to stderr, the rest to stdout.
- Progress bars go to stdout, start with ``\\r`` and look like
  ``  |=========>      | 3/8 - 1.23s/it`` (``=`` bars: sampling, VAE tiling) or
  ``  |#########       | 120/800 - 512.00MB/s`` (``#`` bars: model loading).

The format is not a stable API. Unknown text is ignored, so a format change only
loses progress detail, never the job.
"""

from __future__ import annotations

import re
from collections import deque

from local_image_mcp.types import ProgressEvent, ProgressStage

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BAR = re.compile(r"\|(?P<bar>[=#> ]*)\|\s*(?P<step>\d+)/(?P<total>\d+)\s+-")
_LOG = re.compile(r"^\[(?P<level>[DVIWE?])\]\s+(?P<msg>.*?)(?:\s+---\s+\S+:\d+)?$")
_VERSION = re.compile(r"version\s+(?P<version>\S+?),?\s+commit\s+(?P<commit>[0-9a-fA-F]+)")

# Log message prefix -> stage that starts with it.
_STAGE_MARKERS: tuple[tuple[str, ProgressStage], ...] = (
    ("generate_image ", ProgressStage.ENCODING),
    ("generating image:", ProgressStage.SAMPLING),
    ("sampling completed", ProgressStage.DECODING),
    ("decoding ", ProgressStage.DECODING),
    ("save result image", ProgressStage.SAVING),
)


def parse_version(text: str) -> tuple[str, str | None]:
    """Return ``(version, commit)`` from the ``sd-cli --version`` output."""
    match = _VERSION.search(text)
    if match is None:
        first = text.strip().splitlines()[0] if text.strip() else "unknown"
        return first, None
    return match["version"], match["commit"].lower()


class SdCliOutputParser:
    """Feed raw output chunks; read progress events and error lines."""

    def __init__(self, total_steps: int, keep_lines: int = 40) -> None:
        self.total_steps = total_steps
        self.stage = ProgressStage.LOADING
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.tail: deque[str] = deque(maxlen=keep_lines)
        self._buffer = ""
        self._last_step: int | None = None

    def feed(self, chunk: str) -> list[ProgressEvent]:
        """Parse a chunk of output. Partial segments wait for the next chunk."""
        self._buffer += chunk
        parts = re.split(r"[\r\n]", self._buffer)
        self._buffer = parts.pop()
        events: list[ProgressEvent] = []
        for part in parts:
            event = self._segment(part)
            if event is not None:
                events.append(event)
        return events

    def finish(self) -> list[ProgressEvent]:
        """Parse what remains in the buffer at the end of the process."""
        rest, self._buffer = self._buffer, ""
        event = self._segment(rest)
        return [event] if event is not None else []

    def error_summary(self) -> str:
        if self.errors:
            return "; ".join(dict.fromkeys(self.errors))
        if self.tail:
            return " | ".join(list(self.tail)[-5:])
        return "no output"

    def _segment(self, raw: str) -> ProgressEvent | None:
        text = _ANSI.sub("", raw).strip()
        if not text:
            return None
        bar = _BAR.search(text)
        if bar is not None:
            return self._bar(bar)
        self.tail.append(text)
        log = _LOG.match(text)
        if log is None:
            return None
        level, message = log["level"], log["msg"]
        if level == "E":
            self.errors.append(message)
        elif level == "W":
            self.warnings.append(message)
        for prefix, stage in _STAGE_MARKERS:
            if message.startswith(prefix) and stage != self.stage:
                self.stage = stage
                if stage is ProgressStage.SAMPLING:
                    self._last_step = 0
                    return ProgressEvent(stage=stage, step=0, total_steps=self.total_steps)
                return ProgressEvent(stage=stage)
        return None

    def _bar(self, match: re.Match[str]) -> ProgressEvent | None:
        if "#" in match["bar"]:  # model loading (bytes), not a sampling step
            return None
        if self.stage not in (ProgressStage.ENCODING, ProgressStage.SAMPLING):
            return None  # for example VAE tiling during decoding
        step, total = int(match["step"]), int(match["total"])
        self.stage = ProgressStage.SAMPLING
        if step == self._last_step:
            return None
        self._last_step = step
        return ProgressEvent(stage=ProgressStage.SAMPLING, step=step, total_steps=total)
