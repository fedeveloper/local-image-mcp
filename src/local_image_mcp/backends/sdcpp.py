"""stable-diffusion.cpp backend: runs ``sd-cli`` as a subprocess, one process per image.

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
}

_TERMINATE_GRACE_S = 5.0
_VERSION_TIMEOUT_S = 15.0
_METADATA_TIMEOUT_S = 30.0
_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)


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
    ) -> None:
        """``command`` replaces the binary with a full command prefix (used by tests).

        ``log_dir``: if set, the raw sd-cli output of each run is saved there
        (``<image stem>.sd-cli.log``); the newest ``keep_logs`` files are kept.
        """
        self._configured = binary
        self._command = list(command) if command is not None else None
        self._health: BackendHealth | None = None
        self.log_dir = log_dir
        self.keep_logs = keep_logs

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

    def _probe(self) -> BackendHealth:
        try:
            prefix = self.command_prefix()
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
