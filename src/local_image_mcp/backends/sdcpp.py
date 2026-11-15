"""stable-diffusion.cpp backend.

Two modes:

- ``cli`` (default): runs ``sd-cli`` as a subprocess, one process per image.
- ``server``: runs one ``sd-server`` process that keeps the model loaded between images
  (see ``sdserver.py``). The server starts at the first job, restarts when the model or the
  load options change, and stops ``idle_unload_s`` seconds after the job queue is empty.

Verified against stable-diffusion.cpp commit ``532d2c1`` (2026-11-15):
``sd-cli --help`` output and ``docs/models/z_image.md`` / ``docs/models/flux2.md``.
Options used: ``--diffusion-model``, ``--llm``, ``--vae``, ``-m/--model``, ``--prompt-file``,
``--negative-prompt-file``, ``-W``, ``-H``, ``--steps``, ``--cfg-scale``, ``-s/--seed``,
``--sampling-method``, ``--scheduler``, ``-r/--ref-image`` (repeatable), ``--offload-to-cpu``,
``--vae-tiling``, ``--backend te=cpu``, ``--flash-attn``, ``-o/--output``, ``--log-level``,
``--version``, ``-M metadata --image <png> --metadata-format json``.
"""

from __future__ import annotations

import asyncio
import base64
import codecs
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO, cast

from local_image_mcp.backends.base import BackendHealth, ProgressCallback
from local_image_mcp.backends.sdcpp_output import SdCliOutputParser, parse_version
from local_image_mcp.backends.sdserver import (
    TERMINAL_STATUSES,
    HttpError,
    SdServerProcess,
    free_port,
    job_error,
    progress_event,
)
from local_image_mcp.errors import (
    BackendNotAvailableError,
    GenerationFailedError,
    InvalidRequestError,
    JobCancelledError,
)
from local_image_mcp.resolver import ResolvedModel
from local_image_mcp.types import BackendOutput, ProgressEvent, ProgressStage, ResolvedSettings

VERIFIED_COMMIT = "532d2c1"
BINARY_NAMES = ("sd-cli", "sd")  # "sd" is the name used by builds before the rename
SERVER_BINARY_NAME = "sd-server"
MODES = ("cli", "server")

# Registry file role -> sd-cli option.
ROLE_OPTIONS: dict[str, str] = {
    "model": "--model",
    "diffusion_model": "--diffusion-model",
    "text_encoder": "--llm",
    "vae": "--vae",
}

# Registry backend_options that this backend understands, with their defaults.
DEFAULT_OPTIONS: dict[str, Any] = {
    "offload_to_cpu": False,
    "vae_tiling": False,
    "text_encoder_on_cpu": False,  # "--backend te=cpu": keeps a large text encoder off the GPU
    "flash_attn": None,  # "on" | "off" | "auto" | module assignments; None = sd-cli default
    "timeout_s": 1800,
    "extra_args": [],
    "server_start_timeout_s": 600,
}

_TERMINATE_GRACE_S = 5.0
_VERSION_TIMEOUT_S = 15.0
_METADATA_TIMEOUT_S = 30.0
_JOB_POLL_S = 0.5
_CANCEL_WAIT_S = 30.0
_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_server_binary(configured: str | None, cli_binary: str | None) -> str | None:
    """Return the configured ``sd-server``, else the one next to ``sd-cli``, else ``PATH``."""
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        return shutil.which(configured)
    if cli_binary:
        cli = Path(cli_binary)
        sibling = cli.with_name(SERVER_BINARY_NAME + cli.suffix)
        if sibling.is_file():
            return str(sibling)
    return shutil.which(SERVER_BINARY_NAME)


def find_binary(configured: str | None) -> str | None:
    """Return the configured binary, else ``sd-cli`` or ``sd`` from ``PATH``."""
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        return shutil.which(configured)
    for name in BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


class SdCppBackend:
    name = "sdcpp"

    def __init__(
        self,
        binary: str | None = None,
        command: Sequence[str] | None = None,
        log_dir: Path | None = None,
        keep_logs: int = 50,
        *,
        mode: str = "cli",
        server_binary: str | None = None,
        server_command: Sequence[str] | None = None,
        idle_unload_s: float = 120.0,
        server_model_dir: Path | None = None,
    ) -> None:
        """``command`` / ``server_command`` replace the binaries with a full command prefix
        (used by tests).

        ``log_dir``: if set, the raw sd-cli output of each run is saved there
        (``<image stem>.sd-cli.log``); the newest ``keep_logs`` files are kept.
        The sd-server output goes to ``<log_dir>/sd-server.log``.
        """
        if mode not in MODES:
            raise InvalidRequestError(f"sdcpp mode must be one of {', '.join(MODES)}.")
        self._configured = binary
        self._command = list(command) if command is not None else None
        self._health: BackendHealth | None = None
        self.log_dir = log_dir
        self.keep_logs = keep_logs
        self.mode = mode
        self._configured_server = server_binary
        self._server_command = list(server_command) if server_command is not None else None
        self.idle_unload_s = idle_unload_s
        self.server_model_dir = server_model_dir
        self._server: SdServerProcess | None = None
        self._server_key: tuple[str, ...] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._server_lock = asyncio.Lock()

    # ------------------------------------------------------------------ health

    def command_prefix(self) -> list[str]:
        if self._command is not None:
            return list(self._command)
        found = find_binary(self._configured)
        if found is None:
            raise BackendNotAvailableError(_not_found_message(self._configured))
        return [found]

    def check(self) -> BackendHealth:
        if self._health is None or not self._health.available:
            self._health = self._probe()
        return self._health

    def server_command_prefix(self) -> list[str]:
        if self._server_command is not None:
            return list(self._server_command)
        cli = find_binary(self._configured) if self._command is None else None
        found = find_server_binary(self._configured_server, cli)
        if found is None:
            raise BackendNotAvailableError(
                "sd-server not found. It is in the same release archive as sd-cli. "
                "Put it next to sd-cli, or set LOCAL_IMAGE_SD_SERVER_BINARY."
            )
        return [found]

    def _probe(self) -> BackendHealth:
        try:
            prefix = (
                self.server_command_prefix() if self.mode == "server" else self.command_prefix()
            )
        except BackendNotAvailableError as exc:
            return BackendHealth(name=self.name, available=False, detail=exc.message)
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [*prefix, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_VERSION_TIMEOUT_S,
                check=False,
                creationflags=_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return BackendHealth(
                name=self.name, available=False, detail=f"cannot run {prefix[0]}: {exc}"
            )
        version, commit = parse_version(done.stdout + done.stderr)
        label = f"{version} (commit {commit})" if commit else version
        detail = None
        if commit is None:
            detail = "could not read the commit from --version"
        elif not VERIFIED_COMMIT.startswith(commit[:7]) and not commit.startswith(VERIFIED_COMMIT):
            detail = f"verified with commit {VERIFIED_COMMIT}; options may differ"
        if self.mode == "server":
            label = f"{label}, sd-server mode"
        return BackendHealth(name=self.name, available=True, version=label, detail=detail)

    # ---------------------------------------------------------------- generate

    def build_argv(
        self,
        settings: ResolvedSettings,
        model: ResolvedModel,
        output_file: Path,
        prompt_file: Path,
        negative_prompt_file: Path | None,
        options: dict[str, Any],
    ) -> list[str]:
        opts = {**DEFAULT_OPTIONS, **options}
        unknown = sorted(set(options) - set(DEFAULT_OPTIONS))
        if unknown:
            raise InvalidRequestError(f"Unknown sdcpp backend option(s): {', '.join(unknown)}")
        argv = self.command_prefix()
        for role, resolved in model.files.items():
            flag = ROLE_OPTIONS.get(role)
            if flag is None:
                known = ", ".join(ROLE_OPTIONS)
                raise InvalidRequestError(
                    f"Model '{model.spec.id}': file role '{role}' is not supported by "
                    f"the sdcpp backend (known roles: {known})."
                )
            argv += [flag, str(resolved.path)]
        argv += ["--prompt-file", str(prompt_file)]
        if negative_prompt_file is not None:
            argv += ["--negative-prompt-file", str(negative_prompt_file)]
        argv += [
            "-W", str(settings.width),
            "-H", str(settings.height),
            "--steps", str(settings.steps),
            "--cfg-scale", _float(settings.cfg_scale),
            "--seed", str(settings.seed),
        ]  # fmt: skip
        if settings.sampling_method:
            argv += ["--sampling-method", settings.sampling_method]
        if settings.scheduler:
            argv += ["--scheduler", settings.scheduler]
        for ref in settings.reference_images:
            argv += ["-r", str(ref)]
        if opts["offload_to_cpu"]:
            argv.append("--offload-to-cpu")
        if opts["vae_tiling"]:
            argv.append("--vae-tiling")
        if opts["text_encoder_on_cpu"]:
            argv += ["--backend", "te=cpu"]
        if opts["flash_attn"]:
            argv += ["--flash-attn", str(opts["flash_attn"])]
        argv += [str(a) for a in opts["extra_args"]]
        argv += ["--log-level", "info", "-o", str(output_file)]
        return argv

    async def generate(
        self,
        settings: ResolvedSettings,
        model: ResolvedModel,
        output_file: Path,
        options: dict[str, Any],
        on_progress: ProgressCallback,
        cancel: asyncio.Event,
    ) -> BackendOutput:
        health = self.check()
        if not health.available:
            raise BackendNotAvailableError(health.detail or "sd-cli not available")
        if self.mode == "server":
            return await self._generate_with_server(
                settings, model, output_file, options, on_progress, cancel, health
            )
        timeout_s = float({**DEFAULT_OPTIONS, **options}["timeout_s"])

        # sd-cli reads "%d" in -o as an image-sequence pattern, and a crash can leave a
        # partial file: write to a safe temporary name, then rename.
        partial = output_file.with_name(f".local-image-{uuid.uuid4().hex}.png")
        with tempfile.TemporaryDirectory(prefix="local-image-") as tmp:
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text(settings.prompt, encoding="utf-8")
            negative_file = None
            if settings.negative_prompt:
                negative_file = Path(tmp) / "negative.txt"
                negative_file.write_text(settings.negative_prompt, encoding="utf-8")
            argv = self.build_argv(settings, model, partial, prompt_file, negative_file, options)
            parser = SdCliOutputParser(settings.steps)
            on_progress(ProgressEvent(stage=ProgressStage.LOADING))
            log_file = self._log_path(output_file)
            raw_log = _RawLog(log_file)
            try:
                code = await _run(argv, parser, on_progress, cancel, timeout_s, raw_log)
                if code != 0 or not partial.is_file():
                    hint = f" Full log: {log_file}" if log_file else ""
                    raise GenerationFailedError(
                        f"sd-cli failed (exit code {code}): {parser.error_summary()}.{hint}"
                    )
                partial.replace(output_file)
            finally:
                raw_log.close()
                with contextlib.suppress(OSError):
                    partial.unlink(missing_ok=True)

        native = await self._read_native_record(output_file)
        return BackendOutput(
            image_path=output_file,
            backend_name=self.name,
            backend_version=health.version or "unknown",
            argv=_portable_argv(argv, partial, output_file),
            options={k: v for k, v in {**DEFAULT_OPTIONS, **options}.items() if v},
            native_record=native,
            log_file=log_file,
        )

    # ------------------------------------------------------------ server mode

    def model_args(self, model: ResolvedModel, options: dict[str, Any]) -> list[str]:
        """The model-loading options, shared by sd-cli and sd-server."""
        opts = {**DEFAULT_OPTIONS, **options}
        unknown = sorted(set(options) - set(DEFAULT_OPTIONS))
        if unknown:
            raise InvalidRequestError(f"Unknown sdcpp backend option(s): {', '.join(unknown)}")
        args: list[str] = []
        for role, resolved in model.files.items():
            flag = ROLE_OPTIONS.get(role)
            if flag is None:
                known = ", ".join(ROLE_OPTIONS)
                raise InvalidRequestError(
                    f"Model '{model.spec.id}': file role '{role}' is not supported by "
                    f"the sdcpp backend (known roles: {known})."
                )
            args += [flag, str(resolved.path)]
        if opts["offload_to_cpu"]:
            args.append("--offload-to-cpu")
        if opts["text_encoder_on_cpu"]:
            args += ["--backend", "te=cpu"]
        if opts["flash_attn"]:
            args += ["--flash-attn", str(opts["flash_attn"])]
        args += [str(a) for a in opts["extra_args"]]
        return args

    def server_argv(self, model: ResolvedModel, options: dict[str, Any], port: int) -> list[str]:
        argv = [*self.server_command_prefix(), *self.model_args(model, options)]
        if self.server_model_dir is not None:
            self.server_model_dir.mkdir(parents=True, exist_ok=True)
            argv += ["--model-dir", str(self.server_model_dir)]
        argv += ["--listen-ip", "127.0.0.1", "--listen-port", str(port), "--log-level", "info"]
        return argv

    @property
    def server_running(self) -> bool:
        return self._server is not None and self._server.running

    async def _ensure_server(
        self,
        model: ResolvedModel,
        options: dict[str, Any],
        on_progress: ProgressCallback,
    ) -> SdServerProcess:
        key = tuple([*self.server_command_prefix(), *self.model_args(model, options)])
        if self._server is not None and self._server.running and self._server_key == key:
            return self._server
        await self._stop_server()
        on_progress(ProgressEvent(stage=ProgressStage.LOADING, message="starting sd-server"))
        port = free_port()
        log_file = self.log_dir / "sd-server.log" if self.log_dir is not None else None
        server = SdServerProcess(self.server_argv(model, options, port), port, log_file)
        timeout = float({**DEFAULT_OPTIONS, **options}["server_start_timeout_s"])
        await server.start(timeout)
        self._server, self._server_key = server, key
        return server

    async def _stop_server(self) -> None:
        server, self._server, self._server_key = self._server, None, None
        if server is not None:
            await server.stop()

    def _cancel_idle_timer(self) -> None:
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    async def on_queue_empty(self) -> None:
        """Called by the job queue when no job is waiting: stop sd-server after a delay."""
        if self._server is None:
            return
        self._cancel_idle_timer()
        if self.idle_unload_s <= 0:
            await self._locked_stop()
            return

        async def stop_later() -> None:
            await asyncio.sleep(self.idle_unload_s)
            # Once the stop begins, a new job must not interrupt it half way.
            await asyncio.shield(self._locked_stop())

        self._idle_task = asyncio.create_task(stop_later(), name="sd-server-idle")

    async def _locked_stop(self) -> None:
        async with self._server_lock:
            await self._stop_server()

    async def aclose(self) -> None:
        """Stop sd-server now (server shutdown)."""
        self._cancel_idle_timer()
        await self._stop_server()

    def _server_payload(
        self, settings: ResolvedSettings, options: dict[str, Any]
    ) -> dict[str, Any]:
        opts = {**DEFAULT_OPTIONS, **options}
        sample: dict[str, Any] = {
            "sample_steps": settings.steps,
            "guidance": {"txt_cfg": settings.cfg_scale},
        }
        if settings.sampling_method:
            sample["sample_method"] = settings.sampling_method
        if settings.scheduler:
            sample["scheduler"] = settings.scheduler
        payload: dict[str, Any] = {
            "prompt": settings.prompt,
            "negative_prompt": settings.negative_prompt or "",
            "width": settings.width,
            "height": settings.height,
            "seed": settings.seed,
            "batch_count": 1,
            "sample_params": sample,
            "ref_images": [
                base64.b64encode(p.read_bytes()).decode("ascii") for p in settings.reference_images
            ],
            "embed_image_metadata": True,
            "preview": "none",
            "output_format": "png",
        }
        if opts["vae_tiling"]:
            payload["vae_tiling_params"] = {"enabled": True}
        return payload

    async def _generate_with_server(
        self,
        settings: ResolvedSettings,
        model: ResolvedModel,
        output_file: Path,
        options: dict[str, Any],
        on_progress: ProgressCallback,
        cancel: asyncio.Event,
        health: BackendHealth,
    ) -> BackendOutput:
        timeout_s = float({**DEFAULT_OPTIONS, **options}["timeout_s"])
        self._cancel_idle_timer()
        async with self._server_lock:
            on_progress(ProgressEvent(stage=ProgressStage.LOADING))
            server = await self._ensure_server(model, options, on_progress)
            payload = self._server_payload(settings, options)
            try:
                submitted = await server.request("POST", "/sdcpp/v1/img_gen", payload)
            except (HttpError, OSError) as exc:
                raise GenerationFailedError(f"sd-server rejected the job: {exc}") from exc
            job_id = str(cast(dict[str, Any], submitted)["id"])
            job = await self._poll_job(server, job_id, on_progress, cancel, timeout_s)

        if job.get("status") == "cancelled":
            raise JobCancelledError("Job cancelled.")
        if job.get("status") != "completed":
            hint = f" Log: {server.log_file}" if server.log_file else ""
            raise GenerationFailedError(f"sd-server job failed: {job_error(job)}.{hint}")
        image = cast(dict[str, Any], cast(dict[str, Any], job["result"])["images"][0])
        partial = output_file.with_name(f".local-image-{uuid.uuid4().hex}.png")
        try:
            partial.write_bytes(base64.b64decode(str(image["b64_json"])))
            on_progress(ProgressEvent(stage=ProgressStage.SAVING))
            partial.replace(output_file)
        finally:
            with contextlib.suppress(OSError):
                partial.unlink(missing_ok=True)
        info: object = image.get("generation_info")
        native = _str_keys(cast(dict[Any, Any], info)) if isinstance(info, dict) else None
        argv = [*server.argv[:1], *self.model_args(model, options)]
        return BackendOutput(
            image_path=output_file,
            backend_name=self.name,
            backend_version=health.version or "unknown",
            argv=[*argv, "POST /sdcpp/v1/img_gen"],
            options={k: v for k, v in {**DEFAULT_OPTIONS, **options}.items() if v}
            | {"mode": "server"},
            native_record=native,
            log_file=server.log_file,
        )

    async def _poll_job(
        self,
        server: SdServerProcess,
        job_id: str,
        on_progress: ProgressCallback,
        cancel: asyncio.Event,
        timeout_s: float,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        cancel_sent_at: float | None = None
        last: ProgressEvent | None = None
        while True:
            if not server.running:
                await self._stop_server()
                raise GenerationFailedError(
                    f"sd-server stopped during the job (exit code {server.returncode})."
                    + (f" Log: {server.log_file}" if server.log_file else "")
                )
            try:
                job = cast(dict[str, Any], await server.request("GET", f"/sdcpp/v1/jobs/{job_id}"))
            except (HttpError, OSError) as exc:
                raise GenerationFailedError(f"cannot read the sd-server job: {exc}") from exc
            if job.get("status") in TERMINAL_STATUSES:
                return job
            event = progress_event(job)
            if event is not None and event != last:
                on_progress(event)
                last = event
            now = loop.time()
            if (cancel.is_set() or now > deadline) and cancel_sent_at is None:
                cancel_sent_at = now
                with contextlib.suppress(HttpError, OSError):
                    await server.request("POST", f"/sdcpp/v1/jobs/{job_id}/cancel")
            if cancel_sent_at is not None and now - cancel_sent_at > _CANCEL_WAIT_S:
                await self._stop_server()  # the job did not stop: free the GPU the hard way
                if cancel.is_set():
                    raise JobCancelledError("Job cancelled.")
                raise GenerationFailedError(f"sd-server did not finish within {timeout_s:.0f} s.")
            await asyncio.sleep(_JOB_POLL_S)

    def _log_path(self, output_file: Path) -> Path | None:
        if self.log_dir is None:
            return None
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logs = sorted(self.log_dir.glob("*.sd-cli.log"), key=lambda p: p.stat().st_mtime)
            for old in logs[: max(len(logs) - self.keep_logs + 1, 0)]:
                old.unlink(missing_ok=True)
        except OSError:
            return None
        return self.log_dir / f"{output_file.stem}.sd-cli.log"

    async def _read_native_record(self, image: Path) -> dict[str, Any] | None:
        """Read the ``sdcpp.generation/v1`` record that sd-cli embeds in the PNG."""
        argv = [*self.command_prefix(), "-M", "metadata", "--image", str(image)]
        argv += ["--metadata-format", "json"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=_NO_WINDOW,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), _METADATA_TIMEOUT_S)
        except (OSError, TimeoutError):
            return None
        try:
            data: object = json.loads(out.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return _str_keys(cast(dict[Any, Any], data))
        return {"metadata": data}


async def _run(
    argv: list[str],
    parser: SdCliOutputParser,
    on_progress: ProgressCallback,
    cancel: asyncio.Event,
    timeout_s: float,
    raw_log: _RawLog,
) -> int:
    """Run sd-cli, stream its output into ``parser``, honour cancel and timeout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=_NO_WINDOW,
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert proc.stdout is not None and proc.stderr is not None

    async def pump(stream: asyncio.StreamReader, name: str) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := await stream.read(4096):
            text = decoder.decode(chunk)
            raw_log.write(text)
            for event in parser.feed(text, name):
                on_progress(event)

    pumps = asyncio.gather(pump(proc.stdout, "stdout"), pump(proc.stderr, "stderr"))
    waiter = asyncio.ensure_future(proc.wait())
    cancelled = asyncio.ensure_future(cancel.wait())
    try:
        done, _ = await asyncio.wait(
            {waiter, cancelled}, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
        if waiter not in done:
            await _stop(proc)
            if cancelled in done:
                raise JobCancelledError("Job cancelled.")
            raise GenerationFailedError(f"sd-cli did not finish within {timeout_s:.0f} s.")
        await pumps
        for event in parser.finish():
            on_progress(event)
        return waiter.result()
    except asyncio.CancelledError:
        await _stop(proc)
        raise
    finally:
        cancelled.cancel()
        if proc.returncode is None:
            await _stop(proc)
        pumps.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pumps


class _RawLog:
    """Best-effort copy of the raw sd-cli output, for troubleshooting."""

    def __init__(self, path: Path | None) -> None:
        self._fh: TextIO | None = None
        if path is not None:
            with contextlib.suppress(OSError):
                self._fh = path.open("w", encoding="utf-8", newline="")

    def write(self, text: str) -> None:
        if self._fh is not None:
            with contextlib.suppress(OSError):
                self._fh.write(text)

    def close(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(OSError):
                self._fh.close()
            self._fh = None


async def _stop(proc: asyncio.subprocess.Process) -> None:
    """Terminate the process (frees its VRAM); kill it if it does not stop."""
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), _TERMINATE_GRACE_S)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


def _float(value: float) -> str:
    return f"{value:g}"


def _portable_argv(argv: list[str], partial: Path, final: Path) -> list[str]:
    """The argv for the metadata: final output name, no temporary prompt file paths."""
    out: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--prompt-file", "--negative-prompt-file") and i + 1 < len(argv):
            out += [arg, "<prompt>" if arg == "--prompt-file" else "<negative prompt>"]
            skip_next = True
        elif arg == str(partial):
            out.append(str(final))
        else:
            out.append(arg)
    return out


def _str_keys(data: dict[Any, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in data.items()}


def _not_found_message(configured: str | None) -> str:
    if configured:
        return (
            f"sd-cli not found at '{configured}' (LOCAL_IMAGE_SD_BINARY). "
            "Check the path to sd-cli.exe / sd-cli."
        )
    exe = "sd-cli.exe" if sys.platform == "win32" else "sd-cli"
    return (
        f"{exe} not found on PATH. Install stable-diffusion.cpp and set "
        "LOCAL_IMAGE_SD_BINARY to the full path of sd-cli. See docs/backends/sdcpp.md."
    )
