"""A managed ``sd-server`` process and its native async HTTP API.

Verified against stable-diffusion.cpp commit ``532d2c1`` (2026-11-15):
``examples/server/README.md``, ``examples/server/api.md``, ``examples/server/main.cpp``,
``examples/server/runtime.cpp`` and ``examples/server/routes_sdcpp.cpp``.

- ``sd-server`` accepts the same model-loading options as ``sd-cli`` (``--diffusion-model``,
  ``--llm``, ``--vae``, ``--offload-to-cpu``, ``--backend``, ...) and loads that model
  **before** it starts to listen. When ``GET /sdcpp/v1/models/status`` answers, the model
  is ready.
- ``--listen-ip`` (default ``127.0.0.1``), ``--listen-port`` (default ``1234``),
  ``--model-dir`` (default ``./models``) and ``--log-level``.
- ``POST /sdcpp/v1/img_gen`` returns ``202`` with a job ``id``. Request fields are merged
  onto the startup defaults, so a partial body is valid. Images are base64 strings.
- ``GET /sdcpp/v1/jobs/{id}``: ``status`` is ``queued | generating | completed | failed |
  cancelled``; ``progress.stage`` is ``queued | waiting | preparing | conditioning |
  sampling | vae_encode | vae_decode | encoding | ...`` with ``completed``/``total`` steps
  during sampling; ``result.images[i].b64_json`` and ``generation_info`` when completed.
- ``POST /sdcpp/v1/jobs/{id}/cancel`` interrupts at the next checkpoint.
- The server has no shutdown endpoint: the client terminates the process.

The HTTP client is the standard library (``urllib``) with proxies disabled, because the
server always listens on the loopback interface.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO, cast

from local_image_mcp.errors import BackendNotAvailableError, GenerationFailedError
from local_image_mcp.types import ProgressEvent, ProgressStage

HOST = "127.0.0.1"
_START_POLL_S = 0.25
_HTTP_TIMEOUT_S = 30.0
_TERMINATE_GRACE_S = 5.0
_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# sd-server job stage -> our stage. Unknown stages keep the last known stage.
STAGE_MAP: dict[str, ProgressStage] = {
    "queued": ProgressStage.LOADING,
    "waiting": ProgressStage.LOADING,
    "preparing": ProgressStage.LOADING,
    "conditioning": ProgressStage.ENCODING,
    "vae_encode": ProgressStage.ENCODING,
    "sampling": ProgressStage.SAMPLING,
    "vae_decode": ProgressStage.DECODING,
    "encoding": ProgressStage.SAVING,
}
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class HttpError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def free_port() -> int:
    """Return a TCP port on the loopback interface that is free now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def _request(url: str, method: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)  # noqa: S310 - loopback http
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with _opener.open(req, timeout=timeout) as resp:
            body = cast(BinaryIO, resp).read()
    except urllib.error.HTTPError as exc:
        raise HttpError(exc.code, exc.read().decode("utf-8", errors="replace")) from exc
    text = body.decode("utf-8", errors="replace")
    return json.loads(text) if text.strip() else None


class SdServerProcess:
    """One ``sd-server`` process with one model loaded."""

    def __init__(self, argv: Sequence[str], port: int, log_file: Path | None) -> None:
        self.argv = list(argv)
        self.port = port
        self.log_file = log_file
        self.base_url = f"http://{HOST}:{port}"
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: BinaryIO | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self._proc is None else self._proc.poll()

    async def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        return await asyncio.to_thread(_request, url, method, payload, _HTTP_TIMEOUT_S)

    async def start(self, timeout_s: float) -> None:
        """Start the process and wait until the model is loaded and the API answers."""
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_file.open("ab")
        sink: Any = self._log if self._log is not None else subprocess.DEVNULL
        try:
            self._proc = subprocess.Popen(  # noqa: S603, ASYNC220 - fixed argv; Popen returns at once
                self.argv,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                creationflags=_NO_WINDOW,
                env={**os.environ, "NO_COLOR": "1"},
            )
        except OSError as exc:
            self._close_log()
            raise BackendNotAvailableError(f"cannot start {self.argv[0]}: {exc}") from exc
        deadline = time.monotonic() + timeout_s
        while True:
            code = self.returncode
            if code is not None:
                self._close_log()
                raise GenerationFailedError(
                    f"sd-server stopped while it loaded the model (exit code {code})."
                    + self._log_hint()
                )
            with contextlib.suppress(OSError, HttpError, ValueError):
                raw = await asyncio.to_thread(
                    _request, self.base_url + "/sdcpp/v1/models/status", "GET", None, 2.0
                )
                status = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
                state = status.get("state")
                if state == "ready":
                    return
                if state == "error":
                    await self.stop()
                    raise GenerationFailedError(
                        f"sd-server could not load the model: {status.get('error')}."
                        + self._log_hint()
                    )
            if time.monotonic() > deadline:
                await self.stop()
                raise GenerationFailedError(
                    f"sd-server did not become ready within {timeout_s:.0f} s." + self._log_hint()
                )
            await asyncio.sleep(_START_POLL_S)

    async def stop(self) -> None:
        """Terminate the process (kill it after a grace period) and free the GPU."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, _TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait)
        self._proc = None
        self._close_log()

    def _close_log(self) -> None:
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
            self._log = None

    def _log_hint(self) -> str:
        return f" Log: {self.log_file}" if self.log_file is not None else ""


def progress_event(job: dict[str, Any]) -> ProgressEvent | None:
    """Convert the ``progress`` object of an sd-server job into a ``ProgressEvent``."""
    raw: object = job.get("progress")
    if not isinstance(raw, dict):
        return None
    progress = cast(dict[str, Any], raw)
    stage = STAGE_MAP.get(str(progress.get("stage", "")))
    if stage is None:
        return None
    if stage is ProgressStage.SAMPLING:
        done, total = progress.get("completed"), progress.get("total")
        if isinstance(done, int) and isinstance(total, int) and total > 0:
            return ProgressEvent(stage=stage, step=done, total_steps=total)
    return ProgressEvent(stage=stage)


def job_error(job: dict[str, Any]) -> str:
    raw: object = job.get("error")
    if isinstance(raw, dict):
        err = cast(dict[str, Any], raw)
        return str(err.get("message") or err.get("code") or "unknown error")
    return str(raw or "unknown error")


__all__ = [
    "STAGE_MAP",
    "TERMINAL_STATUSES",
    "HttpError",
    "SdServerProcess",
    "free_port",
    "job_error",
    "progress_event",
]
