"""MCP server: five tools over stdio, backed by the single-worker job queue.

Every tool returns quickly (see PLAN.md §6). ``generate_image`` validates the request and
queues a job; Claude then calls ``get_job_status`` (with ``wait_seconds`` up to 45) and
``get_job_result``. The tools return file paths, not image bytes, unless the caller asks
for a small preview.

Verified against the official Python SDK ``mcp`` 2.3.0 (``mcp.server.mcpserver.MCPServer``).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver import Image as McpImage
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, ContentBlock, ImageContent, TextContent, ToolAnnotations
from PIL import Image
from pydantic import BaseModel, Field

from local_image_mcp import __version__
from local_image_mcp.config import Settings
from local_image_mcp.errors import LocalImageError
from local_image_mcp.jobs.gpu_lock import GpuLock
from local_image_mcp.jobs.model import Job, JobState, utc_now
from local_image_mcp.jobs.queue import JobQueue
from local_image_mcp.jobs.store import JobStore
from local_image_mcp.service import ImageService, ModelInfo
from local_image_mcp.types import GenerationRequest

log = logging.getLogger(__name__)

SERVER_NAME = "local-image"
MAX_WAIT_SECONDS = 45.0
"""Upper limit of ``wait_seconds``. It stays below 60 s, so no client timeout applies."""
PROGRESS_INTERVAL_S = 2.0
PREVIEW_MAX_SIZE = 384
PREVIEW_JPEG_QUALITY = 80
ENABLE_FAKE_ENV = "LOCAL_IMAGE_ENABLE_FAKE"

INSTRUCTIONS = """\
Generate images on this computer's GPU with local models.

Workflow:
1. Call list_models once. Use only a model with available=true.
2. Call generate_image. It returns a job_id at once.
3. Call get_job_status with wait_seconds=45 until state is succeeded, failed or cancelled.
   One image takes about 20-60 seconds.
4. Call get_job_result to get the image path and its metadata.
   Set include_preview=true only when you must see the image.

Write prompts as natural-language descriptions. Read prompt_notes in list_models.
Images are saved as PNG with a JSON sidecar that can reproduce them.
Do not generate images of real people without their consent.
"""


# --------------------------------------------------------------------------- outputs


class MissingFileInfo(BaseModel):
    role: str
    source: str
    alternatives: list[str] = Field(default_factory=list[str])


class ModelSummary(BaseModel):
    id: str
    name: str
    capabilities: list[str]
    available: bool
    unavailable_reason: str | None = None
    missing_files: list[MissingFileInfo] = Field(default_factory=list[MissingFileInfo])
    license: str
    license_url: str | None = None
    vram_estimate_gb: float | None = None
    supports_negative_prompt: bool
    max_reference_images: int
    defaults: dict[str, Any]
    limits: dict[str, Any]
    prompt_notes: str | None = None


class ListModelsResult(BaseModel):
    models: list[ModelSummary]


class SettingsSummary(BaseModel):
    model: str
    width: int
    height: int
    seed: int
    steps: int
    cfg_scale: float
    negative_prompt: str | None = None
    reference_images: list[str] = Field(default_factory=list[str])
    output_path: str


class GenerateResult(BaseModel):
    job_id: str
    state: str
    queue_position: int | None
    settings: SettingsSummary
    warnings: list[str]
    next_step: str


class JobStatusResult(BaseModel):
    job_id: str
    state: str
    stage: str | None
    step: int | None
    total_steps: int | None
    percent: float | None
    queue_position: int | None
    elapsed_s: float
    message: str | None = None
    error: str | None = None
    warnings: list[str]
    next_step: str | None = None


class JobResultInfo(BaseModel):
    job_id: str
    image_path: str
    metadata_path: str
    model: str
    prompt: str
    seed: int
    width: int | None
    height: int | None
    steps: int | None
    cfg_scale: float | None
    duration_s: float
    backend: str
    warnings: list[str]


# ---------------------------------------------------------------------------- server


class LocalImageApp:
    """The state behind the tools: the service and the job queue."""

    def __init__(self, service: ImageService, queue: JobQueue, *, include_fake: bool) -> None:
        self.service = service
        self.queue = queue
        self.include_fake = include_fake

    @classmethod
    def from_settings(cls, settings: Settings, *, include_fake: bool = False) -> LocalImageApp:
        service = ImageService(settings)
        queue = JobQueue(service, JobStore(settings.jobs_dir), GpuLock(settings.gpu_lock_path))
        return cls(service, queue, include_fake=include_fake)

    # ---------------------------------------------------------------- helpers

    def _visible(self, info: ModelInfo) -> bool:
        return self.include_fake or info.spec.backend != "fake"

    def _check_model_visible(self, model_id: str) -> None:
        spec = self.service.registry.get(model_id)
        if spec is not None and spec.backend == "fake" and not self.include_fake:
            visible = sorted(
                m.id for m in self.service.registry.models.values() if m.backend != "fake"
            )
            known = ", ".join(visible) or "none"
            raise ToolError(f"Unknown model '{model_id}'. Available models: {known}.")

    # ------------------------------------------------------------------ tools

    def list_models(self) -> ListModelsResult:
        summaries: list[ModelSummary] = []
        for info in self.service.list_models():
            if not self._visible(info):
                continue
            spec, avail = info.spec, info.availability
            reason: str | None = None
            if not info.backend_available:
                reason = f"Backend '{spec.backend}' is not available: {info.backend_detail}"
            elif not avail.available:
                reason = "Model files are missing. Run: local-image models pull " + spec.id
            summaries.append(
                ModelSummary(
                    id=spec.id,
                    name=spec.display_name,
                    capabilities=[str(c) for c in spec.capabilities],
                    available=info.backend_available and avail.available,
                    unavailable_reason=reason,
                    missing_files=[
                        MissingFileInfo(role=m.role, source=m.source, alternatives=m.alternatives)
                        for m in avail.missing
                    ],
                    license=spec.license,
                    license_url=spec.license_url,
                    vram_estimate_gb=spec.vram_estimate_gb,
                    supports_negative_prompt=spec.supports_negative_prompt,
                    max_reference_images=spec.max_reference_images,
                    defaults=spec.defaults.model_dump(exclude_none=True),
                    limits=spec.limits.model_dump(exclude_none=True),
                    prompt_notes=spec.prompt_notes,
                )
            )
        return ListModelsResult(models=summaries)

    def generate(self, request: GenerationRequest) -> GenerateResult:
        self._check_model_visible(request.model)
        try:
            job = self.queue.submit(request)
        except LocalImageError as exc:
            raise ToolError(exc.message) from exc
        prepared = self.queue.prepared(job.id)
        assert prepared is not None  # the job was just submitted by this process
        s = prepared.settings
        return GenerateResult(
            job_id=job.id,
            state=str(job.state),
            queue_position=self.queue.queue_position(job.id),
            settings=SettingsSummary(
                model=s.model,
                width=s.width,
                height=s.height,
                seed=s.seed,
                steps=s.steps,
                cfg_scale=s.cfg_scale,
                negative_prompt=s.negative_prompt,
                reference_images=[str(p) for p in s.reference_images],
                output_path=str(prepared.output_file),
            ),
            warnings=list(job.warnings),
            next_step=f"Call get_job_status with job_id='{job.id}' and wait_seconds=45.",
        )

    def _get(self, job_id: str) -> Job:
        try:
            return self.queue.get(job_id)
        except LocalImageError as exc:
            raise ToolError(exc.message) from exc

    def status(self, job: Job, now: datetime | None = None) -> JobStatusResult:
        now = now or utc_now()
        start = job.started_at or job.created_at
        end = job.finished_at or now
        p = job.progress
        next_step: str | None
        if job.state is JobState.SUCCEEDED:
            next_step = f"Call get_job_result with job_id='{job.id}'."
        elif job.state.is_terminal:
            next_step = None
        else:
            next_step = f"Call get_job_status again with job_id='{job.id}' and wait_seconds=45."
        return JobStatusResult(
            job_id=job.id,
            state=str(job.state),
            stage=str(p.stage) if p is not None else None,
            step=p.step if p is not None else None,
            total_steps=p.total_steps if p is not None else None,
            percent=job.percent,
            queue_position=self.queue.queue_position(job.id),
            elapsed_s=round(max((end - start).total_seconds(), 0.0), 1),
            message=p.message if p is not None else None,
            error=job.error,
            warnings=list(job.warnings),
            next_step=next_step,
        )

    async def wait_status(
        self, job_id: str, wait_seconds: float, ctx: Context | None = None
    ) -> JobStatusResult:
        job = self._get(job_id)
        wait = min(max(wait_seconds, 0.0), MAX_WAIT_SECONDS)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while not job.state.is_terminal:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            job = await self.queue.wait(job_id, timeout=min(remaining, PROGRESS_INTERVAL_S))
            if ctx is not None and not job.state.is_terminal:
                await _report(ctx, job)
        return self.status(job)

    def result(self, job_id: str, include_preview: bool) -> CallToolResult:
        job = self._get(job_id)
        if job.state is not JobState.SUCCEEDED or job.result is None:
            if job.state.is_terminal:
                detail = f": {job.error}" if job.error else "."
                raise ToolError(f"Job '{job_id}' has no image. It ended as {job.state}{detail}")
            raise ToolError(
                f"Job '{job_id}' is still {job.state}. Call get_job_status with wait_seconds=45."
            )
        r, req = job.result, job.request
        prepared = self.queue.prepared(job_id)
        settings = prepared.settings if prepared is not None else None
        info = JobResultInfo(
            job_id=job.id,
            image_path=str(r.image_path),
            metadata_path=str(r.metadata_path),
            model=req.model,
            prompt=req.prompt,
            seed=r.seed,
            width=settings.width if settings else req.width,
            height=settings.height if settings else req.height,
            steps=settings.steps if settings else req.steps,
            cfg_scale=settings.cfg_scale if settings else req.cfg_scale,
            duration_s=round(r.duration_s, 1),
            backend=f"{r.backend_name} {r.backend_version}".strip(),
            warnings=list(r.warnings),
        )
        content: list[ContentBlock] = [TextContent(type="text", text=info.model_dump_json())]
        if include_preview:
            try:
                content.append(make_preview(r.image_path))
            except OSError as exc:
                content.append(TextContent(type="text", text=f"No preview: {exc}"))
        return CallToolResult(content=content, structured_content=info.model_dump(mode="json"))

    async def cancel(self, job_id: str) -> JobStatusResult:
        try:
            job = await self.queue.cancel(job_id)
        except LocalImageError as exc:
            raise ToolError(exc.message) from exc
        return self.status(job)


async def _report(ctx: Context, job: Job) -> None:
    """Send a progress notification. A failure must not break the tool call."""
    p = job.progress
    stage = str(p.stage) if p is not None else str(job.state)
    try:
        if p is not None and p.step is not None and p.total_steps:
            await ctx.report_progress(p.step, p.total_steps, stage)
        else:
            await ctx.report_progress(0, None, stage)
    except Exception:  # the client may not ask for progress
        log.debug("progress notification failed", exc_info=True)


def make_preview(image_path: Path) -> ImageContent:
    """Return a small JPEG of the image (longest side ``PREVIEW_MAX_SIZE``)."""

    with Image.open(image_path) as img:
        img.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE))
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=PREVIEW_JPEG_QUALITY)

    return McpImage(data=buffer.getvalue(), format="jpeg").to_image_content()


def create_server(app: LocalImageApp) -> MCPServer[None]:
    """Build the MCP server around ``app``. The lifespan starts and stops the job queue."""

    @asynccontextmanager
    async def lifespan(_server: MCPServer[None]) -> AsyncGenerator[None]:
        await app.queue.start()
        try:
            yield
        finally:
            await app.queue.stop()

    server: MCPServer[None] = MCPServer(
        SERVER_NAME,
        title="Local image generation",
        instructions=INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
    )

    # All four hints are explicit on every tool: some directories reject a tool
    # when a hint is missing. No tool reaches the network: openWorld is always false.
    read_only = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    )
    # generate_image writes new files and never overwrites (see paths.py); every
    # call starts a new job, so it is not idempotent.
    creates_job = ToolAnnotations(
        read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
    )
    # cancel_job stops a job and deletes no file; a second call has no more effect.
    cancels_job = ToolAnnotations(
        read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    )

    @server.tool(annotations=read_only)
    def list_models() -> ListModelsResult:
        """List the local image models, their capabilities, defaults and availability.

        Use a model only when available is true. missing_files names the exact files to
        download with `local-image models pull <id>`.
        """
        return app.list_models()

    @server.tool(annotations=creates_job)
    def generate_image(
        prompt: Annotated[str, Field(description="What the image shows, in natural language.")],
        model: Annotated[str, Field(description="A model id from list_models.")],
        width: Annotated[
            int | None,
            Field(
                description="Pixels. Default: model default, or the shape of the first "
                "reference image when reference_images is set and no size is given."
            ),
        ] = None,
        height: Annotated[int | None, Field(description="Pixels. Default: model default.")] = None,
        seed: Annotated[int | None, Field(description="Same seed = same image.")] = None,
        steps: Annotated[int | None, Field(description="Default: model default.")] = None,
        cfg_scale: Annotated[
            float | None,
            Field(description="Guidance; 1.0 means no CFG. Default: model default."),
        ] = None,
        negative_prompt: Annotated[
            str | None, Field(description="Only for models with supports_negative_prompt.")
        ] = None,
        reference_images: Annotated[
            list[str] | None,
            Field(
                description="Paths of input images (png, jpg, webp, bmp), for models with "
                "'reference' or 'edit'. Relative paths start at the project directory. "
                "Describe the change in the prompt."
            ),
        ] = None,
        output_path: Annotated[
            str | None,
            Field(description="PNG path or directory. Default: <project>/generated-images/."),
        ] = None,
    ) -> GenerateResult:
        """Start one image generation job and return its job_id at once.

        The request is validated first, so errors come back immediately. Then call
        get_job_status with wait_seconds=45 until the job ends.
        """
        request = GenerationRequest(
            prompt=prompt,
            model=model,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            negative_prompt=negative_prompt,
            reference_images=tuple(Path(p) for p in reference_images or ()),
            output_path=output_path,
        )
        return app.generate(request)

    @server.tool(annotations=read_only)
    async def get_job_status(
        job_id: str,
        ctx: Context,
        wait_seconds: Annotated[
            float,
            Field(ge=0, le=MAX_WAIT_SECONDS, description="Wait up to this time for the end."),
        ] = 0,
    ) -> JobStatusResult:
        """Return the state and progress of a job.

        With wait_seconds > 0 the call returns when the job ends or the time passes.
        """
        return await app.wait_status(job_id, wait_seconds, ctx)

    @server.tool(annotations=read_only)
    def get_job_result(
        job_id: str,
        include_preview: Annotated[
            bool, Field(description="Add a small JPEG preview (about 384 px).")
        ] = False,
    ) -> CallToolResult:
        """Return the image path, the metadata path and the settings of a finished job."""
        return app.result(job_id, include_preview)

    @server.tool(annotations=cancels_job)
    async def cancel_job(job_id: str) -> JobStatusResult:
        """Cancel a queued or running job. A finished job does not change."""
        return await app.cancel(job_id)

    return server


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``local-image-mcp``: run the server over stdio."""
    del argv
    logging.basicConfig(
        level=os.environ.get("LOCAL_IMAGE_LOG_LEVEL", "WARNING").upper(),
        stream=sys.stderr,  # stdout carries the MCP protocol
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    include_fake = os.environ.get(ENABLE_FAKE_ENV, "") not in ("", "0", "false")
    app = LocalImageApp.from_settings(settings, include_fake=include_fake)
    create_server(app).run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["LocalImageApp", "create_server", "main"]
