# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- All five MCP tools declare the four annotation hints (`readOnlyHint`,
  `destructiveHint`, `idempotentHint`, `openWorldHint`) as explicit booleans, so
  hosts can decide when to ask the user for confirmation.
- Server mode for the `sdcpp` backend: one `sd-server`
  process keeps the model loaded between images, restarts when the model changes, and
  stops after an idle grace period (`LOCAL_IMAGE_SD_SERVER_IDLE`, default 120 s).
- `LOCAL_IMAGE_SD_MODE`: `auto` (default; server mode when `sd-server` is found),
  `cli` or `server`.
- `local-image doctor` shows the sd mode and the `sd-server` path.
- `LOCAL_IMAGE_QUANT`: preferred quantization when several are in the cache.
- `local-image models pull <model> --quant <Q>`: download a chosen quantization.
- `local-image models init-registry`: create a commented user registry with an example.
- Docs: [docs/models.md](docs/models.md) (other quantizations, files and models).

### Tested

- Server mode on Windows 11, RTX 5060 8 GB, sd.cpp `5dcfa79`: Z-Image-Turbo 1024×1024
  takes about 14 s per image with the loaded model (22 s with the model load).

### Removed

- PyPI release workflow (`release.yml`) and `docs/releasing.md`: the project is not
  published on PyPI. Install it from a clone.

## 0.1.0rc1 - 2026-11-15

First release candidate.

### Added

- Core: model registry (YAML, user overrides), Hugging Face cache resolver with
  quantization and text-encoder alternatives and sharded safetensors support, output
  path policy, JSON sidecar metadata that can reproduce each image.
- Single-worker job queue with a cross-process GPU lock and persistent job store.
- `sdcpp` backend for stable-diffusion.cpp `sd-cli`: safe prompt files, progress
  parsing, cancel and timeout, raw logs, version check (verified commit `532d2c1`).
- Models: `z-image-turbo` (Z-Image-Turbo GGUF) and `flux2-klein-4b`
  (FLUX.2 klein 4B GGUF, up to 4 reference images).
- MCP server `local-image-mcp` (stdio) with `list_models`, `generate_image`,
  `get_job_status` (long-poll up to 45 s, progress notifications), `get_job_result`
  (optional small JPEG preview) and `cancel_job`.
- CLI `local-image`: `doctor`, `models list|check|pull`, `generate`, `regenerate`,
  `claude install-skill|show-skill|claude-md`.
- Claude Code integration: the `local-image` skill and a CLAUDE.md section template.
- Docs: Windows and Linux install, Claude Code guide, backend guide, responsible use.

### Tested

- Windows 11, RTX 5060 8 GB, sd-cli `5dcfa79` (CUDA 12 release): Z-Image-Turbo
  1024×1024 in 22–36 s (~5.3 GB VRAM); FLUX.2 klein 4B 1024×1024 in 22 s (~3.6 GB);
  a reference edit in 26 s (~4.6 GB); image generation from a Claude Code session.

