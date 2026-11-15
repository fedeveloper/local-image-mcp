"""``local-image`` command line interface, on top of the same core as the MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any, TextIO

from huggingface_hub import constants as hf_constants

from local_image_mcp import __version__, claude_code
from local_image_mcp.backends.sdcpp import SdCppBackend, find_binary, find_server_binary
from local_image_mcp.config import Settings, default_user_registry
from local_image_mcp.errors import LocalImageError
from local_image_mcp.jobs import GpuLock, Job, JobQueue, JobState, JobStore
from local_image_mcp.metadata import read_sidecar
from local_image_mcp.registry import FileRef
from local_image_mcp.service import ImageService, ModelInfo
from local_image_mcp.types import GenerationRequest, ProgressEvent, ProgressStage

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CANCELLED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-image", description="Generate images with local diffusion models."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="list, check and download models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    p = models_sub.add_parser("list", help="list the models in the registry")
    p.add_argument("--json", action="store_true", help="print JSON")
    p = models_sub.add_parser("check", help="show which model files are in the local cache")
    p.add_argument("model", nargs="?", help="model id (default: all models)")
    p.add_argument("--json", action="store_true", help="print JSON")
    p = models_sub.add_parser("pull", help="download the missing files of a model")
    p.add_argument("model", help="model id")
    p.add_argument(
        "--quant", help="quantization to download, for example Q8_0 (default: the first file)"
    )
    p = models_sub.add_parser(
        "init-registry", help="create a user registry file with comments and examples"
    )
    p.add_argument("--force", action="store_true", help="replace an existing file")

    gen = sub.add_parser("generate", help="generate one image")
    gen.add_argument("-p", "--prompt", required=True)
    gen.add_argument("-m", "--model", required=True)
    gen.add_argument("-W", "--width", type=int)
    gen.add_argument("-H", "--height", type=int)
    gen.add_argument("-s", "--seed", type=int)
    gen.add_argument("--steps", type=int)
    gen.add_argument("--cfg-scale", type=float)
    gen.add_argument("-n", "--negative-prompt")
    gen.add_argument(
        "-r", "--ref", action="append", default=[], metavar="IMAGE", help="reference image"
    )
    gen.add_argument("-o", "--output", help="output .png file or directory")
    gen.add_argument("--json", action="store_true", help="print the result as JSON")

    regen = sub.add_parser("regenerate", help="generate an image again from its metadata JSON")
    regen.add_argument("metadata", type=Path)
    regen.add_argument("-o", "--output", help="output .png file or directory")
    regen.add_argument("--json", action="store_true", help="print the result as JSON")

    sub.add_parser("doctor", help="show configuration and backend status")

    claude = sub.add_parser("claude", help="Claude Code integration files")
    claude_sub = claude.add_subparsers(dest="claude_command", required=True)
    p = claude_sub.add_parser("install-skill", help="install the local-image skill")
    p.add_argument(
        "--scope",
        choices=["user", "project"],
        default="user",
        help="user: ~/.claude/skills (default); project: <project>/.claude/skills",
    )
    p.add_argument("--force", action="store_true", help="replace a changed skill file")
    claude_sub.add_parser("show-skill", help="print the SKILL.md")
    claude_sub.add_parser("claude-md", help="print a section to paste into CLAUDE.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.command == "claude":
            return _claude(args, settings)
        if args.command == "models" and args.models_command == "init-registry":
            return _models_init_registry(settings, force=args.force)
        service = ImageService(settings)
        return _dispatch(args, settings, service)
    except LocalImageError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return EXIT_CANCELLED


def _dispatch(args: argparse.Namespace, settings: Settings, service: ImageService) -> int:
    command: str = args.command
    if command == "models":
        sub: str = args.models_command
        if sub == "list":
            return _models_list(service, as_json=args.json)
        if sub == "check":
            return _models_check(service, args.model, as_json=args.json)
        return _models_pull(service, args.model, args.quant)
    if command == "generate":
        request = GenerationRequest(
            prompt=args.prompt,
            model=args.model,
            width=args.width,
            height=args.height,
            seed=args.seed,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            negative_prompt=args.negative_prompt,
            reference_images=tuple(Path(r) for r in args.ref),
            output_path=args.output,
        )
        return asyncio.run(_generate(settings, service, request, None, as_json=args.json))
    if command == "regenerate":
        meta = read_sidecar(args.metadata)
        request = meta.to_request(output_path=args.output)
        return asyncio.run(
            _generate(settings, service, request, meta.file_overrides(), as_json=args.json)
        )
    return _doctor(settings, service)


# ---------------------------------------------------------------- claude


def _claude(args: argparse.Namespace, settings: Settings) -> int:
    sub: str = args.claude_command
    if sub == "show-skill":
        print(claude_code.skill_text(), end="")
        return EXIT_OK
    if sub == "claude-md":
        print(claude_code.claude_md_section(), end="")
        return EXIT_OK
    scope = claude_code.SkillScope(args.scope)
    target = claude_code.skill_path(scope, settings.project_dir)
    if claude_code.install_skill(target, force=args.force):
        print(f"Installed the skill: {target}")
        print("Start a new Claude Code session to load it.")
    else:
        print(f"The skill is up to date: {target}")
    return EXIT_OK


# ---------------------------------------------------------------- models


def _models_list(service: ImageService, *, as_json: bool) -> int:
    infos = service.list_models()
    if as_json:
        _print_json([_model_summary(i) for i in infos])
        return EXIT_OK
    for info in infos:
        spec = info.spec
        status = "ready" if info.availability.available and info.backend_available else "not ready"
        print(f"{spec.id}  [{status}]")
        print(f"    {spec.display_name}")
        print(f"    capabilities: {', '.join(c.value for c in spec.capabilities)}")
        print(f"    license: {spec.license}   backend: {spec.backend}", end="")
        if spec.vram_estimate_gb is not None:
            print(f"   VRAM (est.): {spec.vram_estimate_gb} GB", end="")
        print()
    return EXIT_OK


def _models_check(service: ImageService, model_id: str | None, *, as_json: bool) -> int:
    specs = [service.model(model_id)] if model_id else list(service.registry.models.values())
    results = [service.resolver.status(spec) for spec in specs]
    if as_json:
        _print_json([r.model_dump(mode="json") for r in results])
    else:
        for result in results:
            print(
                f"{result.model_id}: {'all files found' if result.available else 'MISSING FILES'}"
            )
            for f in result.files:
                print(f"    ok       {f.role}: {f.path}")
            for m in result.missing:
                extra = f" (or {', '.join(m.alternatives)})" if m.alternatives else ""
                print(f"    missing  {m.role}: {m.source}{extra}")
    return EXIT_OK if all(r.available for r in results) else EXIT_ERROR


def _models_pull(service: ImageService, model_id: str, quant: str | None) -> int:
    spec = service.model(model_id)
    resolved = service.resolver.pull(
        spec,
        quant=quant,
        on_file=lambda role, src: print(f"downloading {role}: {src}", file=sys.stderr),
    )
    for role, f in resolved.files.items():
        print(f"{role}: {f.path}")
    if quant and quant != service.resolver.prefer_quant:
        print(
            f"note: if other quantizations of '{model_id}' are in the cache too, set "
            f"LOCAL_IMAGE_QUANT={quant} to use this one.",
            file=sys.stderr,
        )
    return EXIT_OK


USER_TEMPLATE = "user_models_template.yaml"


def _models_init_registry(settings: Settings, *, force: bool) -> int:
    target = settings.user_registry or default_user_registry()
    if target.exists() and not force:
        print(f"error: {target} exists. Use --force to replace it.", file=sys.stderr)
        return EXIT_ERROR
    target.parent.mkdir(parents=True, exist_ok=True)
    text = resources.files("local_image_mcp.registry").joinpath(USER_TEMPLATE).read_text("utf-8")
    target.write_text(text, encoding="utf-8")
    print(f"created {target}")
    print("Edit it, then run: local-image models check")
    return EXIT_OK


def _model_summary(info: ModelInfo) -> dict[str, Any]:
    spec = info.spec
    return {
        "id": spec.id,
        "display_name": spec.display_name,
        "backend": spec.backend,
        "capabilities": [c.value for c in spec.capabilities],
        "license": spec.license,
        "license_url": spec.license_url,
        "vram_estimate_gb": spec.vram_estimate_gb,
        "defaults": spec.defaults.model_dump(mode="json"),
        "files_available": info.availability.available,
        "missing_files": [m.model_dump(mode="json") for m in info.availability.missing],
        "backend_available": info.backend_available,
        "backend_detail": info.backend_detail,
    }


# --------------------------------------------------------------- generate


async def _generate(
    settings: Settings,
    service: ImageService,
    request: GenerationRequest,
    overrides: dict[str, FileRef] | None,
    *,
    as_json: bool,
) -> int:
    queue = JobQueue(service, JobStore(settings.jobs_dir), GpuLock(settings.gpu_lock_path))
    reporter = _ProgressPrinter(sys.stderr)
    queue.add_listener(lambda _job, event: reporter.show(event))
    async with queue:
        job = queue.submit(request, overrides)
        for warning in job.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        try:
            job = await queue.wait(job.id)
        except asyncio.CancelledError:
            await queue.cancel(job.id)
            raise
        reporter.close()
    return _report(job, as_json=as_json)


def _report(job: Job, *, as_json: bool) -> int:
    if as_json:
        _print_json(json.loads(job.model_dump_json()))
    elif job.state is JobState.SUCCEEDED and job.result is not None:
        print(job.result.image_path)
        print(f"metadata: {job.result.metadata_path}", file=sys.stderr)
        print(f"seed: {job.result.seed}   time: {job.result.duration_s:.1f} s", file=sys.stderr)
    else:
        print(f"error: job {job.state.value}: {job.error}", file=sys.stderr)
    if job.state is JobState.SUCCEEDED:
        return EXIT_OK
    return EXIT_CANCELLED if job.state is JobState.CANCELLED else EXIT_ERROR


class _ProgressPrinter:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.last: ProgressEvent | None = None
        self.tty = stream.isatty()

    def show(self, event: ProgressEvent) -> None:
        if event == self.last:
            return
        self.last = event
        if event.stage is ProgressStage.SAMPLING and event.step and event.total_steps:
            text = f"sampling {event.step}/{event.total_steps}"
        else:
            text = event.stage.value.replace("_", " ")
        if self.tty:
            self.stream.write(f"\r\033[K{text}")
        else:
            self.stream.write(f"{text}\n")
        self.stream.flush()

    def close(self) -> None:
        if self.tty:
            self.stream.write("\r\033[K")
            self.stream.flush()


# ----------------------------------------------------------------- doctor


def _doctor(settings: Settings, service: ImageService) -> int:
    print(f"local-image-mcp {__version__}")
    print(f"python:          {sys.version.split()[0]}")
    print(f"project dir:     {settings.project_dir}")
    print(f"output dir:      {settings.output_dir}")
    print(f"allowed roots:   {', '.join(str(r) for r in settings.allowed_roots)}")
    print(f"state dir:       {settings.state_dir}")
    registry_hint = f"(none; local-image models init-registry creates {default_user_registry()})"
    print(f"user registry:   {settings.user_registry or registry_hint}")
    print(f"HF cache:        {settings.hf_cache_dir or hf_constants.HF_HUB_CACHE}")
    print(f"preferred quant: {settings.quant or '(none: registry order)'}")
    cli_binary = find_binary(settings.sd_binary)
    print(f"sd-cli:          {cli_binary or '(not found)'}")
    sdcpp_backend = service.backends.get("sdcpp")
    effective = (
        sdcpp_backend.effective_mode()
        if isinstance(sdcpp_backend, SdCppBackend)
        else settings.sd_mode
    )
    mode_label = f"auto ({effective})" if settings.sd_mode == "auto" else settings.sd_mode
    print(f"sd mode:         {mode_label}")
    if effective == "server":
        server_binary = find_server_binary(settings.sd_server_binary, cli_binary)
        print(f"sd-server:       {server_binary or '(not found)'}")
        print(f"sd-server idle:  {settings.sd_server_idle_s:g} s")
    print("backends:")
    for name, backend in service.backends.items():
        health = backend.check()
        state = "available" if health.available else "not available"
        detail = f" ({health.detail})" if health.detail else ""
        version = f" {health.version}" if health.version else ""
        print(f"    {name}: {state}{version}{detail}")
    return EXIT_OK


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))
