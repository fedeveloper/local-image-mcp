"""The image service: validation, defaults, file resolution and one generation run.

Both the CLI and the MCP server use this class (through ``JobQueue``).
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from local_image_mcp.backends import Backend, ProgressCallback, create_backends
from local_image_mcp.config import Settings
from local_image_mcp.errors import (
    BackendNotAvailableError,
    CapabilityError,
    InvalidRequestError,
    UnknownModelError,
)
from local_image_mcp.metadata import (
    BackendRecord,
    FileRecord,
    ImageMetadata,
    ModelRecord,
    request_record,
    write_sidecar,
)
from local_image_mcp.paths import (
    OutputPolicy,
    check_reference_image,
    sidecar_path,
    unique_output_path,
)
from local_image_mcp.registry import FileRef, ModelSpec, Registry, load_registry
from local_image_mcp.resolver import ModelAvailability, ModelResolver, ResolvedModel
from local_image_mcp.types import (
    Capability,
    GenerationRequest,
    GenerationResult,
    ProgressEvent,
    ProgressStage,
    ResolvedSettings,
)

MAX_SEED = 2**63 - 1
MAX_PROMPT_CHARS = 8000


class PreparedJob(BaseModel):
    """A validated request, ready to run."""

    model_config = ConfigDict(frozen=True)

    settings: ResolvedSettings
    model: ResolvedModel
    output_file: Path
    options: dict[str, Any]
    warnings: list[str]


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: ModelSpec
    availability: ModelAvailability
    backend_available: bool
    backend_detail: str | None


class ImageService:
    def __init__(
        self,
        settings: Settings,
        *,
        registry: Registry | None = None,
        resolver: ModelResolver | None = None,
        backends: dict[str, Backend] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry if registry is not None else load_registry(settings.user_registry)
        self.resolver = resolver if resolver is not None else ModelResolver(settings.hf_cache_dir)
        self.backends = backends if backends is not None else create_backends(settings)
        self.policy = OutputPolicy(
            settings.project_dir, settings.output_dir, settings.allowed_roots
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    # ----------------------------------------------------------------- queries

    def model(self, model_id: str) -> ModelSpec:
        spec = self.registry.get(model_id)
        if spec is None:
            known = ", ".join(sorted(self.registry.models)) or "none"
            raise UnknownModelError(f"Unknown model '{model_id}'. Available models: {known}.")
        return spec

    def backend(self, name: str) -> Backend:
        backend = self.backends.get(name)
        if backend is None:
            raise BackendNotAvailableError(f"Backend '{name}' is not registered.")
        return backend

    def list_models(self) -> list[ModelInfo]:
        infos: list[ModelInfo] = []
        for spec in self.registry.models.values():
            health = self.backend(spec.backend).check()
            infos.append(
                ModelInfo(
                    spec=spec,
                    availability=self.resolver.status(spec),
                    backend_available=health.available,
                    backend_detail=health.detail,
                )
            )
        return infos

    # -------------------------------------------------------------- validation

    def prepare(
        self, request: GenerationRequest, file_overrides: dict[str, FileRef] | None = None
    ) -> PreparedJob:
        """Validate a request and fill in defaults. Raises a ``LocalImageError`` subclass."""
        spec = self.model(request.model)
        warnings: list[str] = []

        prompt = request.prompt.strip()
        if not prompt:
            raise InvalidRequestError("The prompt is empty.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise InvalidRequestError(f"The prompt is longer than {MAX_PROMPT_CHARS} characters.")

        references = self._check_references(spec, request.reference_images)

        negative = (request.negative_prompt or "").strip() or None
        if negative is not None and not spec.supports_negative_prompt:
            warnings.append(
                f"Model '{spec.id}' does not use a negative prompt; it was ignored. "
                "Write constraints in the prompt instead."
            )
            negative = None

        d, lim = spec.defaults, spec.limits
        width = request.width if request.width is not None else d.width
        height = request.height if request.height is not None else d.height
        for name, value in (("width", width), ("height", height)):
            if not lim.min_size <= value <= lim.max_size:
                raise InvalidRequestError(
                    f"{name} {value} is outside {lim.min_size}..{lim.max_size} for '{spec.id}'."
                )
            if value % lim.size_multiple:
                raise InvalidRequestError(
                    f"{name} {value} must be a multiple of {lim.size_multiple} for '{spec.id}'."
                )
        if lim.max_pixels is not None and width * height > lim.max_pixels:
            raise InvalidRequestError(
                f"{width}x{height} is more than {lim.max_pixels} pixels, "
                f"the limit for '{spec.id}' on this setup."
            )

        steps = request.steps if request.steps is not None else d.steps
        if not lim.min_steps <= steps <= lim.max_steps:
            raise InvalidRequestError(
                f"steps {steps} is outside {lim.min_steps}..{lim.max_steps} for '{spec.id}'."
            )

        cfg_scale = request.cfg_scale if request.cfg_scale is not None else d.cfg_scale
        if not 0.0 <= cfg_scale <= 30.0:
            raise InvalidRequestError(f"cfg_scale {cfg_scale} is outside 0..30.")

        seed = request.seed if request.seed is not None else secrets.randbelow(2**32)
        if not 0 <= seed <= MAX_SEED:
            raise InvalidRequestError(f"seed {seed} is outside 0..{MAX_SEED}.")

        health = self.backend(spec.backend).check()
        if not health.available:
            raise BackendNotAvailableError(
                f"Backend '{spec.backend}' for model '{spec.id}' is not available: "
                f"{health.detail or 'unknown reason'}"
            )

        resolved = self.resolver.resolve(spec, file_overrides)
        output_file = self.policy.output_file(request.output_path, spec.id, seed, self._clock())

        settings = ResolvedSettings(
            prompt=prompt,
            model=spec.id,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            negative_prompt=negative,
            sampling_method=d.sampling_method,
            scheduler=d.scheduler,
            reference_images=references,
        )
        return PreparedJob(
            settings=settings,
            model=resolved,
            output_file=output_file,
            options=dict(spec.backend_options),
            warnings=warnings,
        )

    def _check_references(self, spec: ModelSpec, images: tuple[Path, ...]) -> tuple[Path, ...]:
        if not images:
            return ()
        if not (spec.has(Capability.REFERENCE) or spec.has(Capability.EDIT)):
            capable = sorted(
                set(self.registry.with_capability(Capability.REFERENCE))
                | set(self.registry.with_capability(Capability.EDIT))
            )
            hint = f" Models with this capability: {', '.join(capable)}." if capable else ""
            raise CapabilityError(f"Model '{spec.id}' does not support reference images.{hint}")
        if len(images) > spec.max_reference_images:
            raise InvalidRequestError(
                f"Model '{spec.id}' accepts at most {spec.max_reference_images} "
                f"reference images, got {len(images)}."
            )
        return tuple(check_reference_image(p, self.settings.project_dir) for p in images)

    # --------------------------------------------------------------------- run

    async def run(
        self,
        prepared: PreparedJob,
        on_progress: ProgressCallback,
        cancel: asyncio.Event,
    ) -> GenerationResult:
        """Run one prepared job on its backend and write the sidecar metadata."""
        spec = prepared.model.spec
        backend = self.backend(spec.backend)
        # Another job may have used the name since prepare(): pick the final name now.
        output_file = unique_output_path(prepared.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        output = await backend.generate(
            prepared.settings, prepared.model, output_file, prepared.options, on_progress, cancel
        )
        duration = round(time.monotonic() - started, 3)

        on_progress(ProgressEvent(stage=ProgressStage.SAVING))
        meta_path = sidecar_path(output.image_path)
        metadata = ImageMetadata(
            created_at=self._clock(),
            duration_s=duration,
            image_file=output.image_path.name,
            request=request_record(prepared.settings),
            model=ModelRecord(
                registry_id=spec.id,
                display_name=spec.display_name,
                license=spec.license,
                files={
                    role: FileRecord.from_resolved(f) for role, f in prepared.model.files.items()
                },
            ),
            backend=BackendRecord(
                name=output.backend_name,
                version=output.backend_version,
                argv=output.argv,
                options=output.options,
                native_record=output.native_record,
            ),
            warnings=prepared.warnings,
        )
        write_sidecar(meta_path, metadata)
        return GenerationResult(
            image_path=output.image_path,
            metadata_path=meta_path,
            seed=prepared.settings.seed,
            duration_s=duration,
            backend_name=output.backend_name,
            backend_version=output.backend_version,
            warnings=prepared.warnings,
        )
