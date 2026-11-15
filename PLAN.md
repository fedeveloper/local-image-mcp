# PLAN.md — local-image-mcp

Status: **Plan approved on 2026-11-15. Phase 1 done (see §13). Next: Phase 2.** Decisions are in §12.

This plan describes an open source MCP server. The server lets Claude Code (and other MCP
clients) generate images with GGUF diffusion models that run on a local GPU.
The reasoning stays in the Anthropic model. The pixels come from the local GPU.

---

## 1. Facts verified in Phase 0

Each fact below was checked on 2026-11-15. The plan depends on these facts.
Section 9 lists the facts that still need a check on the real Windows machine.

### 1.1 stable-diffusion.cpp

Source: `leejet/stable-diffusion.cpp`, commit `532d2c1` (2026-11-15). Release tags use the
form `master-<short-sha>`.

- **The executable is now `sd-cli`, not `sd`.** The repository also builds `sd-server`.
  Docs: `examples/cli/README.md`, `README.md`. Our backend accepts a configured path and
  looks for `sd-cli` first, then `sd` (older builds) on `PATH`.
- The CLI options that we need exist in the source (`examples/common/*.cpp`):
  `--diffusion-model`, `--llm` (text encoder), `--vae`, `-p/--prompt`, `--prompt-file`,
  `--negative-prompt`, `--cfg-scale`, `--guidance`, `--steps`, `--seed`, `-W/--width`,
  `-H/--height`, `-r/--ref-image`, `--sampling-method`, `--scheduler`, `--offload-to-cpu`,
  `--vae-tiling`, `-o/--output`, `--log-level`, `--disable-image-metadata`, `-M/--mode`.
  The full `sd-cli --help` output is in Appendix A. `-r` can be repeated (several
  reference images). `--steps` defaults to 20 and `--cfg-scale` to 7.0, so the backend
  always passes both explicitly.
- `sd-cli --version` prints `stable-diffusion.cpp version unknown, commit 532d2c1` for a
  source build. The backend parses the commit for the metadata.
- Z-Image-Turbo example from `docs/models/z_image.md`:
  `--diffusion-model <z_image_turbo gguf> --vae ae.sft --llm Qwen3-4B-Instruct-2507-Q4_K_M.gguf
  --cfg-scale 1.0 --steps 8 -H 1024 -W 512 --offload-to-cpu`.
  The doc says that Z-Image runs "on GPUs with 4GB of VRAM — or even less".
- FLUX.2 klein 4B examples from `docs/models/flux2.md`:
  - text-to-image: `--cfg-scale 1.0 --steps 4 --offload-to-cpu`,
    text encoder Qwen3 4B, FLUX.2 VAE.
  - reference / edit: the same plus `-r <image>` and `--sampling-method euler`.
- An omitted `--seed` means a random seed. Our server always passes an explicit seed.
- `sd-cli` embeds its own generation record (`sdcpp.generation/v1` JSON) in the PNG.
  `sd-cli -M metadata --image out.png --metadata-format json` reads it back.
  We copy this record into our sidecar JSON.
- `sd-server` has a native async job API (`POST /sdcpp/v1/img_gen`,
  `GET /sdcpp/v1/jobs/{id}`, `POST /sdcpp/v1/jobs/{id}/cancel`). It keeps the model in
  memory between requests. This is a good candidate for a later backend (see Q4).

### 1.2 Hugging Face repositories and real file names

The diffusion repositories from unsloth contain **only the diffusion model**.
They contain no text encoder and no VAE. The registry must name those files in other
repositories.

| Role | Repository | Example file (real name) | Size | License | Gated |
|---|---|---|---|---|---|
| Z-Image-Turbo diffusion | `unsloth/Z-Image-Turbo-GGUF` | `z-image-turbo-Q4_K_M.gguf` | 3.85 GB | apache-2.0 | no |
| Z-Image text encoder | `unsloth/Qwen3-4B-Instruct-2507-GGUF` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2.50 GB | apache-2.0 | no |
| Z-Image VAE (FLUX.1 VAE) | `Comfy-Org/z_image_turbo` | `split_files/vae/ae.safetensors` | 335 MB | apache-2.0 | no |
| FLUX.2 klein 4B diffusion | `unsloth/FLUX.2-klein-4B-GGUF` | `flux-2-klein-4b-Q4_K_M.gguf` | 2.60 GB | apache-2.0 | no |
| FLUX.2 klein text encoder | `unsloth/Qwen3-4B-GGUF` | `Qwen3-4B-Q4_K_M.gguf` | 2.50 GB | apache-2.0 | no |
| FLUX.2 VAE | `Comfy-Org/flux2-klein-4B` | `split_files/vae/flux2-vae.safetensors` | 336 MB | apache-2.0 | no |

Notes:

- The repository name `unsloth/FLUX.2-klein-4B-GGUF` is correct (verified).
- Available quantizations in both unsloth repos: Q2_K, Q3_K_S, Q3_K_M, Q4_0, Q4_1,
  Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16, BF16.
  File name patterns: `z-image-turbo-<QUANT>.gguf` and `flux-2-klein-4b-<QUANT>.gguf`.
- Z-Image uses **Qwen3-4B-Instruct-2507**. FLUX.2 klein 4B uses **Qwen3-4B** (base).
  These are two different text encoders (source: sd.cpp docs).
- The same VAE files also exist in other repositories:
  `black-forest-labs/FLUX.1-schnell/ae.safetensors` (same size, gated) and
  `black-forest-labs/FLUX.2-dev/ae.safetensors` (same size, gated,
  **flux-non-commercial-license**). The registry uses the non-gated Apache-2.0 copies.
  The resolver can accept alternative sources (see `alternatives` in §4).
- `black-forest-labs/FLUX.2-klein-9B` uses the **flux-non-commercial-license**.
  It is not in the first registry. The README warns about it.
- The model cards (diffusers examples) give these settings:
  Z-Image-Turbo: 1024×1024, `num_inference_steps=9` ("8 DiT forwards"), `guidance_scale=0.0`.
  FLUX.2 klein 4B: 1024×1024, 4 steps.

### 1.3 Claude Code and MCP

Source: `code.claude.com/docs/en/mcp` and `/en/env-vars`.

- `MCP_TIMEOUT` (ms): the **startup** timeout of an MCP server.
- `MCP_TOOL_TIMEOUT` (ms): the wall-clock limit of one tool call. Default: very long
  (about 28 hours when unset). A per-server `"timeout"` in `.mcp.json` overrides it.
  Progress notifications do **not** extend this limit.
- `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT` (ms): a tool call with no response and no progress
  notification aborts after this time. Default: 30 minutes for stdio servers.
- A tool call that runs for more than 2 minutes moves to a background task
  (`CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`, default 120000).
- Output: warning above 10,000 tokens. Default limit 25,000 tokens (`MAX_MCP_OUTPUT_TOKENS`).
- Claude Code sets `CLAUDE_PROJECT_DIR` in the environment of a stdio server.
- Registration syntax: `claude mcp add [options] <name> -- <command> [args...]`,
  with `--scope local|project|user` and `-e KEY=value`.

**Consequence for the design:** every tool of our server returns in well under 60 seconds.
The default timeouts are then never a problem, and no special timeout setting is
necessary. The long work runs in a job. Section 6 documents the optional settings.

### 1.4 Python libraries

- `mcp` (official SDK) is now **2.x** (2.3.0). `FastMCP` was renamed to
  `mcp.server.mcpserver.MCPServer`. Old v1 examples do not work. We pin `mcp>=2.3,<3`.
  `Context.report_progress()` exists.
- `huggingface_hub` 2.4.0 reads `HF_HOME` and `HF_HUB_CACHE`. It offers
  `try_to_load_from_cache()`, `hf_hub_download(local_files_only=True)` and `scan_cache_dir()`.

---

## 2. Goals and non-goals

Goals: the architecture in the brief (core library, swappable backends, YAML registry,
MCP server with async jobs, CLI, reproducible metadata), Windows 11 first, Linux second.

Non-goals for now: training and fine-tuning, cloud generation services, a GUI.
We also do not build `sd-cli` ourselves. The user installs a release binary or builds it.

---

## 3. Repository layout

```
local-image-mcp/
├── pyproject.toml              # uv, ruff, pyright, pytest config; entry points
├── uv.lock
├── LICENSE                     # Apache-2.0
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── PLAN.md
├── .github/
│   └── workflows/ci.yml        # ruff, pyright, pytest (fake backend) on ubuntu + windows
├── src/local_image_mcp/
│   ├── __init__.py             # __version__
│   ├── __main__.py             # python -m local_image_mcp -> CLI
│   ├── config.py               # Settings: env vars + optional TOML config file
│   ├── errors.py               # typed errors with user-facing messages
│   ├── types.py                # GenerationRequest, GenerationResult, Capability, ...
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── schema.py           # pydantic models for the YAML file
│   │   ├── loader.py           # load built-in + user registry, merge, validate
│   │   └── models.yaml         # built-in model registry (package data)
│   ├── resolver.py             # resolve registry entries to local files via huggingface_hub
│   ├── paths.py                # output path policy, file naming, safe path checks
│   ├── metadata.py             # sidecar JSON write/read, schema version, replay
│   ├── jobs/
│   │   ├── __init__.py
│   │   ├── model.py            # Job, JobState, JobProgress
│   │   ├── store.py            # persist jobs as JSON in the state directory
│   │   ├── queue.py            # single-worker async queue, cancel support
│   │   └── gpu_lock.py         # cross-process lock: one GPU job per machine
│   ├── backends/
│   │   ├── __init__.py         # backend factory by name
│   │   ├── base.py             # Backend protocol
│   │   ├── fake.py             # solid-colour PNG, no GPU
│   │   ├── sdcpp.py            # stable-diffusion.cpp via sd-cli subprocess
│   │   ├── sdcpp_progress.py   # parse sd-cli output into progress events
│   │   ├── diffusers.py        # stub: raises BackendNotAvailable
│   │   └── comfyui.py          # stub: raises BackendNotAvailable
│   ├── cli.py                  # `local-image` command (argparse or typer)
│   └── server.py               # MCP server (MCPServer, stdio)
├── integrations/
│   └── claude-code/
│       ├── CLAUDE.md.section.md    # template section for a project CLAUDE.md
│       └── skills/local-image/SKILL.md
├── docs/
│   ├── claude-code.md          # claude mcp add, timeouts, troubleshooting
│   ├── models.md               # registry format, adding a model
│   ├── backends/sdcpp.md       # install sd-cli on Windows and Linux, verified version
│   └── metadata.md             # sidecar JSON schema and replay
├── examples/
│   └── registry.user.yaml      # example of a user registry override
└── tests/
    ├── conftest.py             # fake HF cache fixture, tmp state dir
    ├── test_registry.py
    ├── test_resolver.py
    ├── test_jobs.py
    ├── test_metadata.py
    ├── test_paths.py
    ├── test_cli.py
    ├── test_server.py          # in-memory MCP client session against the server
    ├── test_sdcpp_args.py      # argv building only; no binary needed
    └── test_sdcpp_progress.py  # parser tests with recorded sd-cli output
```

Entry points (`pyproject.toml`):
`local-image = local_image_mcp.cli:main` and `local-image-mcp = local_image_mcp.server:main`.

---

## 4. Model registry (YAML)

The built-in file ships with the package. A user file can add or override entries.
Location of the user file: `LOCAL_IMAGE_REGISTRY` or `<config_dir>/models.yaml`.

```yaml
schema_version: 1
models:
  z-image-turbo:
    display_name: Z-Image-Turbo (Q4_K_M)
    backend: sdcpp
    capabilities: [text2img]
    license: apache-2.0
    license_url: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
    vram_estimate_gb: 7.0          # to verify in Phase 2 on the RTX 5060
    files:
      diffusion_model:
        repo: unsloth/Z-Image-Turbo-GGUF
        filename: z-image-turbo-Q4_K_M.gguf
      text_encoder:
        repo: unsloth/Qwen3-4B-Instruct-2507-GGUF
        filename: Qwen3-4B-Instruct-2507-Q4_K_M.gguf
      vae:
        repo: Comfy-Org/z_image_turbo
        filename: split_files/vae/ae.safetensors
        alternatives:
          - {repo: black-forest-labs/FLUX.1-schnell, filename: ae.safetensors}
    defaults:
      width: 1024
      height: 1024
      steps: 8
      cfg_scale: 1.0               # sd.cpp value; equals diffusers guidance_scale=0
      negative_prompt: null        # not supported: the server rejects it
    limits:
      steps: [1, 16]
      size_multiple: 16            # to verify against sd-cli
      max_pixels: 1572864          # 1536x1024; to tune for 8 GB
    backend_options:
      offload_to_cpu: true

  flux2-klein-4b:
    display_name: FLUX.2 klein 4B (Q4_K_M)
    backend: sdcpp
    capabilities: [text2img, reference, edit]
    license: apache-2.0
    license_url: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
    vram_estimate_gb: 6.0          # to verify in Phase 4
    max_reference_images: 1        # to verify; sd-cli accepts repeated -r
    files:
      diffusion_model:
        repo: unsloth/FLUX.2-klein-4B-GGUF
        filename: flux-2-klein-4b-Q4_K_M.gguf
      text_encoder:
        repo: unsloth/Qwen3-4B-GGUF
        filename: Qwen3-4B-Q4_K_M.gguf
      vae:
        repo: Comfy-Org/flux2-klein-4B
        filename: split_files/vae/flux2-vae.safetensors
    defaults:
      width: 1024
      height: 1024
      steps: 4
      cfg_scale: 1.0
      sampling_method: euler
    backend_options:
      offload_to_cpu: true
```

Validation rules (pydantic): known backend name, known capability names, every file has
`repo` and `filename`, defaults inside limits. A local file path (`path:`) is also allowed
instead of `repo`/`filename`, for users who do not use the HF cache.

---

## 5. Core interfaces

The code below shows the planned shapes. Names can change in review.

```python
# types.py
class Capability(StrEnum):
    TEXT2IMG = "text2img"
    REFERENCE = "reference"
    EDIT = "edit"


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    model_id: str
    width: int | None = None  # None -> registry default
    height: int | None = None
    seed: int | None = None  # None -> random, then recorded
    steps: int | None = None
    cfg_scale: float | None = None
    negative_prompt: str | None = None
    reference_images: tuple[Path, ...] = ()
    output_path: Path | None = None  # file or directory; None -> default output dir


@dataclass(frozen=True)
class ResolvedModel:  # registry entry + absolute local file paths
    spec: ModelSpec
    files: dict[str, ResolvedFile]  # role -> path, repo, filename, revision, size


@dataclass(frozen=True)
class GenerationResult:
    image_path: Path
    metadata_path: Path
    seed: int
    duration_s: float
    backend_name: str
    backend_version: str
    backend_record: dict | None  # e.g. sdcpp.generation/v1 JSON
```

```python
# backends/base.py
class ProgressEvent(NamedTuple):
    stage: Literal["loading", "encoding", "sampling", "decoding", "saving"]
    step: int | None
    total_steps: int | None
    message: str | None


class Backend(Protocol):
    name: str

    def version(self) -> str: ...
    def check(self) -> BackendHealth: ...  # binary found? version? GPU?
    async def generate(
        self,
        request: GenerationRequest,  # already validated and filled with defaults
        model: ResolvedModel,
        output_file: Path,
        on_progress: Callable[[ProgressEvent], None],
        cancel: anyio.Event,
    ) -> GenerationResult: ...
```

```python
# resolver.py
class ModelResolver:
    def __init__(self, cache_dir: Path | None = None, allow_download: bool = False): ...
    def resolve(self, spec: ModelSpec) -> ResolvedModel: ...   # raises MissingModelFiles
    def status(self, spec: ModelSpec) -> ModelAvailability:   # for list_models
```

Resolution order for each file: explicit `path:` → `try_to_load_from_cache(repo, filename)`
(respects `HF_HOME` and `HF_HUB_CACHE`) → each `alternatives` entry → error.
Download happens **only** with `allow_download=True` (CLI flag `local-image models pull`),
and `hf_hub_download` skips files that are already in the cache. The MCP server never
downloads by default, so a tool call never starts a multi-GB transfer.

```python
# jobs/queue.py
class JobQueue:
    async def submit(self, request: GenerationRequest) -> Job: ...  # validates first
    def get(self, job_id: str) -> Job: ...
    async def wait(self, job_id: str, timeout_s: float) -> Job: ...  # long-poll helper
    async def cancel(self, job_id: str) -> Job: ...
```

Job states: `queued → running → succeeded | failed | cancelled`.
One worker runs one job at a time. A cross-process file lock (`filelock` package) on
`<state_dir>/gpu.lock` protects the GPU. Two Claude Code sessions start two server
processes. The lock prevents two jobs from running on the 8 GB GPU at the same time.
Each job is stored as JSON in `<state_dir>/jobs/`. After a restart, finished jobs stay
readable. A job that was `running` at a crash becomes `failed` with a clear message.
`<state_dir>` comes from `platformdirs` (Windows: `%LOCALAPPDATA%\local-image-mcp`).

---

## 6. MCP server

Transport: stdio. SDK: `mcp>=2.3,<3` with `MCPServer`.

| Tool | Input | Output | Returns in |
|---|---|---|---|
| `list_models` | — | id, name, capabilities, license, VRAM estimate, defaults, `available` + missing files | < 1 s |
| `generate_image` | `prompt`, `model`, `width?`, `height?`, `seed?`, `steps?`, `output_path?`, `reference_images?` | `job_id`, queue position, resolved settings | < 1 s |
| `get_job_status` | `job_id`, `wait_seconds?` (0–45) | state, stage, step/total, percent, error, elapsed | ≤ 45 s |
| `get_job_result` | `job_id`, `include_preview?` (default false) | image path, metadata path, seed, metadata summary; optional small JPEG preview as image content | < 2 s |
| `cancel_job` | `job_id` | final state | < 5 s |

Rules:

- `generate_image` validates **before** it queues a job. It rejects:
  an unknown model; `reference_images` for a model without the `reference` capability
  ("Model 'z-image-turbo' does not support reference images. Models with this capability:
  flux2-klein-4b."); a missing model file (with the exact repo and filename);
  sizes outside the limits; a reference image that does not exist; an output path outside
  the allowed roots.
- `wait_seconds` lets Claude wait without many tool calls. The maximum stays below 60 s,
  so no timeout setting is necessary.
- The tools return paths, not image bytes. The optional preview is small
  (for example 384 px JPEG), well below the 10,000-token warning.
- Output path policy: relative paths resolve against `CLAUDE_PROJECT_DIR` (or the current
  directory). Allowed roots: the project directory and the configured output directory.
  Existing files are not overwritten: the server adds a suffix.
- Default file name: `<YYYYMMDD-HHMMSS>_<model>_<seed>.png` plus `.json` with the same stem.

Claude Code registration (to document in `docs/claude-code.md`):

```
claude mcp add --transport stdio --scope user local-image -- uvx local-image-mcp
claude mcp add --transport stdio --scope user -e LOCAL_IMAGE_SD_BINARY=C:\tools\sd\sd-cli.exe local-image -- uvx local-image-mcp
```

Optional settings to document: `MCP_TIMEOUT=30000` if the first start of `uvx` is slow.
No tool timeout setting is necessary, because all tools return quickly.

---

## 7. CLI

Same core, same queue code, but runs one job in the foreground with a progress bar.

```
local-image models list
local-image models check [MODEL]          # which files are in the cache, which are missing
local-image models pull MODEL             # explicit download (only command that downloads)
local-image generate --prompt "..." --model z-image-turbo [--seed 42] [--width 1024]
                     [--height 1024] [--steps 8] [--ref img.png ...] [-o out/]
local-image regenerate path/to/image.json # replay from sidecar metadata
local-image doctor                        # sd-cli path + version, HF cache dir, GPU info
```

---

## 8. Metadata sidecar (JSON)

```json
{
  "schema": "local-image-mcp/metadata/v1",
  "tool_version": "0.1.0",
  "created_at": "2026-11-15T10:54:00Z",
  "duration_s": 41.7,
  "request": {
    "prompt": "...", "negative_prompt": null, "model": "z-image-turbo",
    "width": 1024, "height": 1024, "seed": 123456789, "steps": 8, "cfg_scale": 1.0,
    "sampling_method": null, "reference_images": []
  },
  "model": {
    "registry_id": "z-image-turbo",
    "files": {
      "diffusion_model": {"repo": "unsloth/Z-Image-Turbo-GGUF",
                          "filename": "z-image-turbo-Q4_K_M.gguf",
                          "revision": "<commit hash from the HF cache>", "size": 3850130496}
    }
  },
  "backend": {
    "name": "sdcpp", "version": "master-532d2c1",
    "argv": ["sd-cli", "--diffusion-model", "..."],
    "options": {"offload_to_cpu": true},
    "native_record": { "...": "sdcpp.generation/v1 JSON from the PNG" }
  }
}
```

`regenerate` rebuilds a `GenerationRequest` from `request` and pins the model files
to the recorded repo, filename and revision. It warns when the backend version differs.
Reference images are stored as paths plus SHA-256, so replay can detect a changed input.
Bit-identical output is not guaranteed across GPUs, drivers or backend versions.

---

## 9. Phases

Each phase ends with green `ruff`, `pyright` and `pytest`, and one commit (or one PR).

| Phase | Content | Done when |
|---|---|---|
| 0 | This plan and the questions. | Plan approved. |
| 1 | Project setup (uv, src layout, ruff, pyright, pytest, CI on Ubuntu + Windows). Types, config, registry schema and loader, resolver (with a fake HF cache in tests), paths, metadata, job store + queue + GPU lock, `fake` backend, CLI (`models list/check`, `generate`, `regenerate`). | `local-image generate --model fake-solid ...` writes PNG + JSON. CI green. |
| 2 | `sdcpp` backend: argv builder, `--prompt-file` for safe quoting, progress parser, cancel, version detection, PNG record extraction. Z-Image-Turbo entry. `docs/backends/sdcpp.md`. | You run one Z-Image image on Windows. We record VRAM, time and the real `sd-cli` output for parser tests. |
| 3 | MCP server: five tools, validation, long-poll, preview, job persistence. Tests with an in-memory MCP client and the fake backend. | Claude Code generates an image through the server. |
| 4 | FLUX.2 klein 4B: reference images (`-r`), capability checks, limits for the number of references. | Text-to-image and one reference test on Windows. |
| 5 | Claude Code integration: `CLAUDE.md` section template, `SKILL.md`, `docs/claude-code.md`. | Skill loads and gives correct prompts in a test project. |
| 6 | README (Windows + Linux install, model table, licenses, responsible use), examples, CONTRIBUTING, CHANGELOG, release prep (version, PyPI metadata, tag). | Release candidate ready. |
| 7 | `sd-server` backend: start on first job, keep models loaded while the queue has jobs, stop when the queue is empty. | Several images in a row without a model reload. |

### What I need on the Windows machine

- Phase 2: the output of `sd-cli --help` and `sd-cli --version` **from your binary**, the
  list of files in your HF cache (`hf cache ls` or `local-image models check` from Phase 1),
  and one test run.
- Phase 4: one reference test.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **RTX 5060 (Blackwell, sm_120)** needs CUDA 12.8+. A prebuilt `sd-cli` CUDA binary may not contain sm_120 kernels. | No GPU, or slow first run (PTX JIT). | `local-image doctor` reports the backend in use. Document the CUDA 12.8+ build and the Vulkan build as an alternative. |
| `sd-cli` options change quickly (the rename `sd` → `sd-cli` is one example). | Wrong argv, failed jobs. | Pin a verified version in code and docs. Detect the version at start. Unit tests on the argv builder. |
| Progress output format is not a stable API. | No step progress. | Parser is best-effort with recorded fixtures. Fallback: stage-only progress and elapsed time. |
| Model load time on every call (subprocess loads ~6.5 GB each time). | 10–40 s overhead per image (estimate). | Accept for v1. Later: `sd-server` backend that keeps models loaded (Q4). |
| 8 GB VRAM: diffusion model + text encoder + VAE + activations at 1024². | Out-of-memory. | `--offload-to-cpu` on by default, Q4_K_M by default, `max_pixels` limit, optional `--vae-tiling`. Measure in Phase 2. |
| Several Claude Code sessions → several server processes. | Two jobs on one GPU. | Cross-process GPU lock. |
| Prompt quoting on Windows (quotes, newlines, `%`). | Broken prompts. | Pass the prompt with `--prompt-file` (temp UTF-8 file), never through the shell. |
| Cancel on Windows. | Orphan process holds VRAM. | Terminate the process and wait. Use a Job Object / process group if `sd-cli` starts children (verify). |
| Output path abuse (writing outside the project). | File overwrite. | Allowed roots, no overwrite, path normalization tests. |
| VAE license confusion (same VAE weights exist in non-commercial and Apache repos). | Wrong license claim. | Registry stores the source repo and license per file. README explains it. |
| `mcp` 2.x API is new; many examples on the web are v1. | Wrong code. | Pin `mcp>=2.3,<3`, follow the v2 migration guide, test with the SDK's in-memory client. |
| Z-Image steps: diffusers uses 9 steps for "8 DiT forwards"; sd.cpp docs use `--steps 8`. | Wrong default. | Default 8, compare 8 vs 9 in Phase 2. |

---

## 11. Three alternative names

All three names were free on PyPI on 2026-11-15.

1. **`easel-mcp`** — short and visual. The easel stands next to the artist (Claude).
2. **`darkroom-mcp`** — images develop locally, in your own room.
3. **`localbrush`** — says "local" and "paint". Neutral for non-MCP use of the CLI.

My suggestion: keep `local-image-mcp` for the repository and the PyPI package, because it is
descriptive and easy to find. Use the CLI name `local-image`.

---

## 12. Decisions (answers from Federico, 2026-11-15)

| # | Topic | Decision |
|---|---|---|
| 1 | Executable | `sd-cli.exe`, CUDA build. |
| 2 | Cached files | Q4_K_M quantizations. Text encoders and VAEs are in the HF cache. Phase 2 confirms the exact repositories with `local-image models check`. |
| 3 | Guidance | Public parameter `cfg_scale` with the sd.cpp meaning (`1.0` = no CFG). |
| 4 | `sd-server` backend | Yes, after Phase 4 (new Phase 7). The backend starts `sd-server` when a job arrives, keeps the models loaded while jobs are in the queue, and stops it (frees VRAM) when the queue is empty. |
| 5 | `edit` capability | FLUX.2 klein 4B: `[text2img, reference, edit]`. |
| 6 | Output directory | `<project>/generated-images/`. |
| 7 | Preview | No default. `get_job_result` takes `include_preview` (default `false`); Claude sets it when the user asks to see the result. |
| 8 | Downloads | Only `local-image models pull` downloads. The MCP server never downloads. |
| 9 | Negative prompt | For a model without negative prompt support: ignore it and return a warning. |
| 10 | CI | Matrix on `ubuntu-latest` and `windows-latest`. |
| 11 | Name | Keep `local-image-mcp`. |

Phase 7 (added): `sd-server` backend with the lifecycle in decision 4.

## 13. Phase 1 notes (implementation choices)

- `service.ImageService` holds validation, defaults, file resolution and one run.
  The CLI and (in Phase 3) the MCP server use it through `jobs.JobQueue`.
- The built-in registry also has a `fake` model (backend `fake`, no files) for tests,
  CI and a quick smoke test without a GPU.
- `sdcpp`, `diffusers` and `comfyui` are placeholder backends that report
  "not available". `prepare()` checks the backend before it checks the model files.
- Job records are JSON files in `<state_dir>/jobs/`. Each live process holds
  `<state_dir>/jobs/owners/<pid>.lock`. At start, a queue marks only the unfinished
  jobs of dead processes as failed, so two Claude Code sessions do not break each other.
- Output files: `.png` only. The final name is chosen just before the run, so two
  queued jobs never write the same file. An image or a sidecar with the same name is
  never overwritten.
- Reference images in the sidecar carry a SHA-256, so a replay can detect a changed input.
- The CLI has no extra dependency (argparse). The fake PNG uses only the standard library.

## Appendix A — `sd-cli --help` (commit 532d2c1, CPU build)

I built the binary from source in a Linux container (CPU only) to read the real help text. Your Windows CUDA build may differ if the version differs.

`sd-cli --version` prints: `stable-diffusion.cpp version unknown, commit 532d2c1`.

```text
stable-diffusion.cpp version unknown, commit 532d2c1
Usage: build/bin/sd-cli [options]

CLI Options:
  --input-frames <string>            directory of ordered RGB frames for vid_upscale (use -i for a single image)
  -o, --output <string>              path to write result image to. you can use printf-style %d format specifiers for
                                     image sequences (default: ./output.png) (eg. output_%03d.png). Single-file video
                                     outputs support .avi, .webm, and animated .webp
  --image <string>                   path to the image or video to inspect (for metadata mode)
  --metadata-format <string>         metadata output format, one of [text, json] (default: text)
  --preview-path <string>            path to write preview image to (default: ./preview.png). For image generation, the
                                     filename can have %03d placeholder for sequential numbering. Multi-frame previews
                                     support .avi, .webm, and animated .webp
  --imat-out <string>                compute the imatrix for this run and save it to the provided path
  --upscale-factor <int>             required positive video upscale factor (RelayVSR: 2 or 4; SeedVR2: integer)
  --vsr-window-size <int>            RelayVSR window including both endpoints (default: 16)
  --vsr-temporal-chunk-size <int>    SeedVR2 input frames per independent chunk: 1 or 4n+1 (default: 5)
  --preview-interval <int>           preview interval: in each sampling pass, positive N updates every Nth denoiser step
                                     and -N previews only completed logical step N; 0 previews the final completed step
                                     of the first pass (base-resolution or high-noise). Default: 1
  --output-begin-idx <int>           starting index for output image sequence, must be non-negative (default 0 if
                                     specified %d in output path, 1 otherwise)
  --compression-quality <int>        compression quality of video and JPEG / WebP images (90 by default)
  --convert-name                     convert tensor name (for convert mode)
  --color                            colors the logging tags according to level
  --preview-noisy                    enables previewing noisy inputs of the models rather than the denoised outputs
  --metadata-raw                     include raw hex previews for unparsed metadata payloads
  --metadata-brief                   truncate long metadata text values in text output
  --metadata-all                     include structural/container entries such as IHDR, IDAT, and non-metadata JPEG
                                     segments
  -M, --mode                         run mode, one of [img_gen, adetailer, vid_gen, vid_upscale, upscale, convert,
                                     metadata], default: img_gen
  --preview                          preview method. must be one of the following [none, proj, tae, vae] (default is
                                     none)
  -h, --help                         show this help message and exit
  --imat-in                          load an imatrix file for quantization or continued collection; can be specified
                                     multiple times
  --log-level                        minimum log level, one of [debug, verbose, info, warn, error] (default: info)
  -v, --verbose                      equivalent to --log-level verbose

Context Options:
  -m, --model <string>                     path to full model
  --clip_l <string>                        path to the clip-l text encoder
  --clip_g <string>                        path to the clip-g text encoder
  --clip_vision <string>                   path to the clip-vision encoder
  --t5xxl <string>                         path to the t5xxl text encoder
  --llm <string>                           path to the llm text encoder. For example: (qwenvl2.5 for qwen-image,
                                           mistral-small3.2 for flux2, ...)
  --tokenizer <string>                     tokenizer.json path, or comma-separated main=FILE,clip-l=FILE,clip-g=FILE
                                           assignments; required for PiD and Lens
  --llm_vision <string>                    path to the llm vit
  --qwen2vl <string>                       alias of --llm. Deprecated.
  --qwen2vl_vision <string>                alias of --llm_vision. Deprecated.
  --diffusion-model <string>               path to the standalone diffusion model
  --high-noise-diffusion-model <string>    path to the standalone high noise diffusion model
  --uncond-diffusion-model <string>        path to the standalone unconditional diffusion model, currently used by
                                           Ideogram4 CFG
  --embeddings-connectors <string>         path to embeddings connectors (LTXAV or P-H3-Image)
  --vae <string>                           path to standalone vae model
  --vae-format <string>                    VAE latent format override: auto, flux, sd3, flux2, or wan (default: auto)
  --audio-vae <string>                     path to standalone LTX audio vae model
  --audio-encoder <string>                 path to wav2vec2 audio encoder model (Wan2.2 S2V)
  --taesd <string>                         path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)
  --tae <string>                           alias of --taesd
  --control-net <string>                   path to control net model
  --ip-adapter <string>                    path to IP-Adapter model (requires --clip_vision)
  --motion-module <string>                 path to AnimateDiff motion module (SD 1.5); enables video generation on
                                           --video-frames > 1
  --embd-dir <string>                      embeddings directory
  --lora-model-dir <string>                lora model directory
  --hires-upscalers-dir <string>           highres fix upscaler model directory
  --tensor-type-rules <string>             weight type per tensor pattern (example: "^vae\.=f16,model\.=q8_0")
  --model-args <string>                    extra model args, key=value list. Supports chroma_use_dit_mask,
                                           chroma_use_t5_mask, chroma_t5_mask_pad, qwen_image_zero_cond_t,
                                           qwen_image_2_1_prefix_cache, qwen_image_2_1_prefix_cache_type (auto or a type
                                           name from --type)
  --photo-maker <string>                   path to PHOTOMAKER model
  --pulid-weights <string>                 path to PuLID Flux weights
  --upscale-model <string>                 path to esrgan model.
  --backend <string>                       runtime backend assignment, e.g. cpu or clip=cpu,vae=cuda0,diffusion=vulkan0
  --params-backend <string>                parameter backend assignment, e.g. disk, cpu, or diffusion=disk,clip=cpu
  --split-mode <string>                    weight distribution for modules assigned multiple devices (--backend
                                           "diffusion=cuda0&cuda1"): layer (whole transformer blocks per device,
                                           default) or row (matmul rows split across devices, CUDA only). Accepts a
                                           single mode or per-module assignments, e.g. row or diffusion=row,te=layer
  --rpc-servers <string>                   comma-separated list of RPC servers to connect to for offloading, in the
                                           format host:port, e.g. localhost:50052,192.168.1.3:50052
  --max-vram <string>                      optional per-device budget in GiB for managed weights and runner buffers
                                           during automatic graph-cut execution. Accepts a single value or assignments
                                           by backend/device, e.g. 6 or cuda0=6,vulkan0=4. 0 uses live free VRAM without
                                           an explicit budget; a negative value reserves that much free VRAM
  -t, --threads <int>                      positive number of computation threads; unset uses the library CPU core
                                           selection
  --conditioning-cache-size <int>          maximum cached conditioning results; unset uses the library default, 0
                                           disables caching
  --list-devices                           list available ggml backend devices, including --rpc-servers (one
                                           'name<TAB>description' per line), and exit; the names are the device names
                                           accepted by --backend and --params-backend
  --disable-prefetch                       disable asynchronous next-segment weight prefetch (defaults to false)
  --disable-segmented-compute              force monolithic graph execution even when automatic graph cutting is needed
                                           (defaults to false)
  --eager-load                             load all params into the params backend at model-load time instead of lazily
                                           on first use (defaults to false)
  --force-sdxl-vae-conv-scale              force use of conv scale on sdxl vae
  --offload-to-cpu                         place the weights in RAM to save VRAM, and automatically load them into VRAM
                                           when needed
  --mmap                                   whether to memory-map model
  --control-net-cpu                        deprecated; use --backend controlnet=cpu
  --clip-on-cpu                            deprecated; use --backend te=cpu
  --vae-on-cpu                             deprecated; use --backend vae=cpu
  --sage-attn                              use native CUDA SageAttention in the diffusion model, with flash/default
                                           attention fallback
  --taesd-preview-only                     prevents usage of taesd for decoding the final image; the VAE keeps encoding
                                           / decoding and taesd is only used for step previews
  --linear-scale                           linear input scale override (float, default: 0 = model default, 1 = no
                                           scaling)
  --attn-scale                             flash-attention K/V scale override (float, default: 0 = model default, 1 = no
                                           scaling); applies when flash attention is selected
  --diffusion-conv-direct                  deprecated; use --conv-direct auto,diffusion=on,controlnet=on
  --vae-conv-direct                        deprecated; use --conv-direct auto,vae=on
  --conv-direct                            on|off|auto or diffusion/vae/controlnet assignments (default: auto), e.g.
                                           auto,vae=off. Overrides legacy convolution flags
  --flash-attn                             on|off|auto or module assignments (default: auto), e.g. auto,te=off,vae=on.
                                           Last repeated option wins
  --fa                                     deprecated; use --flash-attn on
  --diffusion-fa                           deprecated; use --flash-attn off,diffusion=on
  --auto-fit                               on|off (default: on). Preserve --backend (otherwise select one GPU) and place
                                           weights on the compute GPU, RAM, another GPU, or disk in that order,
                                           according to available memory (--max-vram limits GPU budgets). Disabled by
                                           explicit --params-backend; uses automatic graph segmentation when needed
  --batched-cfg                            on|off (default: on). Run the conditional and unconditional CFG branches in
                                           one batched UNet forward when supported
  --type                                   weight type (examples: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K,
                                           q4_K). If not specified, the default is the type of the weight file
  --rng                                    RNG, one of [std_default, cuda, cpu], default: cuda(sd-webui), cpu(comfyui)
  --sampler-rng                            sampler RNG, one of [std_default, cuda, cpu]. If not specified, use --rng
  --prediction                             prediction type override, one of [eps, v, edm_v, sd3_flow, flux_flow,
                                           sefi_flow, minit2i_flow, sensenova_u1_flow, flux3_flow, h3_av_flow,
                                           ming_image_flow]
  --lora-apply-mode                        the way to apply LoRA, one of [auto, immediately, at_runtime], default is
                                           auto. In auto mode, if the model weights contain any quantized parameters,
                                           the at_runtime mode will be used; otherwise, immediately will be used.The
                                           immediately mode may have precision and compatibility issues with quantized
                                           parameters, but it usually offers faster inference speed and, in some cases,
                                           lower memory usage. The at_runtime mode, on the other hand, is exactly the
                                           opposite.

Generation Options:
  -p, --prompt <string>                    the prompt to render
  -n, --negative-prompt <string>           the negative prompt (default: "")
  --ad-model <string>                      path to a converted YOLOv8 detection model for ADetailer
  --ad-prompt <string>                     ADetailer prompt; empty inherits the main prompt, supports [PROMPT], [SEP],
                                           and [SKIP]
  --ad-negative-prompt <string>            ADetailer negative prompt; empty inherits the main negative prompt, supports
                                           [PROMPT] and [SEP]
  --extra-ad-args <string>                 extra ADetailer args, key=value list. Supports input_size, confidence, nms,
                                           max_detections, mask_k_largest, mask_min_ratio, mask_max_ratio, dilate_erode,
                                           x_offset, y_offset, mask_mode, merge_masks, invert_mask, mask_blur,
                                           inpaint_padding, inpaint_width, inpaint_height, denoising_strength, steps,
                                           cfg_scale, sample_method, scheduler, sort_by
  -i, --init-img <string>                  path to the init image, or a single input image for vid_upscale
  --end-img <string>                       path to the end image, required by flf2v
  --mask <string>                          path to the mask image
  --control-image <string>                 path to control image, control net
  --ip-adapter-image <string>              path to the IP-Adapter reference image
  --control-video <string>                 path to control video frames, It must be a directory path. The video frames
                                           inside should be stored as images in lexicographical (character) order. For
                                           example, if the control video path is `frames`, the directory contain images
                                           such as 00.png, 01.png, ... etc.
  --pm-id-images-dir <string>              path to PHOTOMAKER input id images dir
  --pm-id-embed-path <string>              path to PHOTOMAKER v2 id embed
  --pulid-id-embedding <string>            path to PuLID id embedding
  --hires-upscaler <string>                highres fix upscaler, Lanczos, Nearest, Latent, Latent (nearest), Latent
                                           (nearest-exact), Latent (antialiased), Latent (bicubic), Latent (bicubic
                                           antialiased), or a model name under --hires-upscalers-dir (default: Latent)
  --control-type <string>                  kind of the control image, one of [openpose, depth, scribble, canny, normal,
                                           segment, tile, repaint]. Required by an SDXL Control Net Union model (tile
                                           and repaint only exist in its ProMax variant), ignored by any other Control
                                           Net. Also accepted: hed, softedge, pidi, ted (scribble); lineart,
                                           anime_lineart, mlsd (canny); seg (segment); inpaint, outpaint (repaint)
  --extra-sample-args <string>             extra sampler/scheduler/guidance args, key=value list. CFG supports
                                           guidance_schedule; APG supports apg_eta, apg_momentum, apg_norm_threshold,
                                           apg_norm_threshold_smoothing; SLG supports slg_uncond; lcm supports
                                           noise_clip_std, noise_scale_start, noise_scale_end; flux supports base_shift,
                                           max_shift; ltx2 supports max_shift, base_shift, stretch, terminal; euler_ge
                                           supports gamma; beta scheduler supports alpha, beta; logit_normal supports
                                           mu, std, logsnr_min, logsnr_max, resolution_aware; llada_image supports
                                           uniform; lms supports lms_max_order, lms_shift, lms_divisions;
                                           noise-injecting samplers support noise_sampler with value iid (default except
                                           for dpm++2m_sde_bt) or brownian_tree; brownian_tree_rng supports cpu
                                           (default), cuda, std_default or sampler_rng
  --extra-tiling-args <string>             extra VAE tiling args, key=value list. Supported video VAEs accept
                                           temporal_tile_frames/temporal_tile_size (default: 4), temporal_tile_overlap
                                           (default: 1); Wan2.2 TI2V 5B accepts temporal_decode_frames (default: 1)
  --ref-image-args <string>                Key-value list to set up the way the reference images are processed (empty =
                                           auto-detect from model weigths)
  --image-preprocess <string>              Image preprocessing rule:
                                           target=init|end|mask|control|ref|ip-adapter|id|control-frame,index=N,mode=aut
                                           o|none|stretch|crop|crop-resize|fit-pad,filter=auto|nearest|nearest-exact|bil
                                           inear|bicubic|lanczos,antialias=auto|true|false,width=W,height=H,anchor=cente
                                           r|top|bottom|left|right,pad_color=#RRGGBB[AA],canny=true|false. Repeat for
                                           multiple rules.
  -H, --height <int>                       image height, in pixel space (default: 512)
  -W, --width <int>                        image width, in pixel space (default: 512)
  --steps <int>                            number of sample steps (default: 20)
  --high-noise-steps <int>                 (high noise) number of sample steps (default: -1 = auto)
  --clip-skip <int>                        ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer
                                           (default: -1). <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
  -b, --batch-count <int>                  batch count
  --qwen-image-layers <int>                number of Qwen Image Layered layers; latent/output count is layers + 1
                                           (default: 3)
  --video-frames <int>                     video frames; unset uses the library default
  --fps <int>                              fps (default: 24)
  --timestep-shift <int>                   shift timestep for NitroFusion models (default: 0). recommended N for
                                           NitroSD-Realism around 250 and 500 for NitroSD-Vibrant
  --upscale-repeats <int>                  Run the ESRGAN upscaler this many times (default: 1)
  --upscale-tile-size <int>                tile size for ESRGAN upscaling (default: 128)
  --hires-width <int>                      highres fix target width, 0 to use --hires-scale (default: 0)
  --hires-height <int>                     highres fix target height, 0 to use --hires-scale (default: 0)
  --hires-steps <int>                      highres fix second pass sample steps, 0 to reuse --steps (default: 0)
  --hires-upscale-tile-size <int>          highres fix upscaler tile size, reserved for model-backed upscalers (default:
                                           128)
  --cfg-scale <float>                      unconditional guidance scale: (default: 7.0)
  --img-cfg-scale <float>                  image guidance scale for inpaint or image edit models: (default: same as
                                           --cfg-scale)
  --guidance <float>                       distilled guidance scale for models with guidance input (default: 3.5)
  --slg-scale <float>                      skip layer guidance (SLG) scale, only for DiT models: (default: 0). 0 means
                                           disabled, a value of 2.5 is nice for sd3.5 medium
  --skip-layer-start <float>               SLG enabling point (default: 0.01)
  --skip-layer-end <float>                 SLG disabling point (default: 0.2)
  --eta <float>                            noise multiplier (default: 0 for ddim_trailing, tcd, res_multistep and
                                           res_2s; 1 for euler_a, er_sde, dpm++2s_a, dpm++2m_sde and dpm++2m_sde_bt)
  --flow-shift <float>                     shift value for Flow models like SD3.x or WAN (default: auto)
  --high-noise-cfg-scale <float>           (high noise) unconditional guidance scale: (default: 7.0)
  --high-noise-img-cfg-scale <float>       (high noise) image guidance scale for inpaint or image edit models (default:
                                           same as --cfg-scale)
  --high-noise-guidance <float>            (high noise) distilled guidance scale for models with guidance input
                                           (default: 3.5)
  --high-noise-slg-scale <float>           (high noise) skip layer guidance (SLG) scale, only for DiT models: (default:
                                           0)
  --high-noise-skip-layer-start <float>    (high noise) SLG enabling point (default: 0.01)
  --high-noise-skip-layer-end <float>      (high noise) SLG disabling point (default: 0.2)
  --high-noise-eta <float>                 (high noise) noise multiplier (default: 0 for ddim_trailing, tcd,
                                           res_multistep and res_2s; 1 for euler_a, er_sde, dpm++2s_a, dpm++2m_sde and
                                           dpm++2m_sde_bt)
  --strength <float>                       strength for noising/unnoising (default: 0.75)
  --pm-style-strength <float>              
  --pulid-id-weight <float>                strength of PuLID identity injection
  --control-strength <float>               strength to apply Control Net (default: 0.9). 1.0 corresponds to full
                                           destruction of information in init image
  --ip-adapter-strength <float>            strength to apply IP-Adapter (default: 1.0)
  --moe-boundary <float>                   timestep boundary for Wan2.2 MoE model. (default: 0.875). Only enabled if
                                           `--high-noise-steps` is set to -1
  --vace-strength <float>                  wan vace strength
  --vae-tile-overlap <float>               tile overlap for vae tiling, in fraction of tile size (default: 0.5)
  --hires-scale <float>                    highres fix scale when target size is not set (default: 2.0)
  --hires-denoising-strength <float>       highres fix second pass denoising strength (default: 0.7)
  --increase-ref-index                     automatically increase the indices of references images based on the order
                                           they are listed (starting with 1).
  --circular                               enable circular padding on both axes for tileable output
  --circularx                              enable circular padding on x-axis (width) only
  --circulary                              enable circular padding on y-axis (height) only
  --disable-image-metadata                 do not embed generation metadata on image files
  --vae-tiling                             process vae encode and decode in spatial tiles to reduce memory usage
                                           (default: 256x256 image pixels)
  --temporal-tiling                        enable temporal tiling for supported video VAE decode
  --hires                                  enable highres fix
  -s, --seed                               RNG seed; unset or a negative value requests a random seed
  --sampling-method                        sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m,
                                           dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd, res_multistep, res_2s,
                                           er_sde, euler_cfg_pp, euler_a_cfg_pp, euler_ge, dpm++2m_sde, dpm++2m_sde_bt,
                                           lms], default: euler for Flux/SD3/Wan, euler_a otherwise
  --high-noise-sampling-method             (high noise) sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a,
                                           dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd, res_multistep,
                                           res_2s, er_sde, euler_cfg_pp, euler_a_cfg_pp, euler_ge, dpm++2m_sde,
                                           dpm++2m_sde_bt, lms], default: euler for Flux/SD3/Wan, euler_a otherwise
  --scheduler                              denoiser sigma scheduler, one of [discrete, karras, exponential, ays, gits,
                                           sgm_uniform, simple, smoothstep, kl_optimal, lcm, bong_tangent, ltx2,
                                           logit_normal, flux2, flux, beta, llada_image], alias: normal=discrete,
                                           default: model-specific
  --sigmas                                 custom sigma values for the sampler, comma-separated (e.g.,
                                           "14.61,7.8,3.5,0.0") or preset: qwen-image-2.1-turbo, p-h3-image,
                                           minimax-h3-image, hyperflow-h3-flash-4, hyperflow-h3-flash-6,
                                           hyperflow-h3-flash-8.
  --hires-sigmas                           custom sigma values for the highres fix second pass, comma-separated (e.g.,
                                           "0.85,0.725,0.421875,0.0") or preset: qwen-image-2.1-turbo, p-h3-image,
                                           minimax-h3-image, hyperflow-h3-flash-4, hyperflow-h3-flash-6,
                                           hyperflow-h3-flash-8.
  --skip-layers                            layers to skip for SLG steps; unset uses the library default
  --high-noise-skip-layers                 (high noise) layers to skip for SLG steps; unset uses the library default
  -r, --ref-image                          reference image for Flux Kontext or MiniMax-H3 Ref2VA (can be used multiple
                                           times)
  --ref-video                              MiniMax-H3 Ref2VA reference video frame directory at 24 fps (can be used
                                           multiple times)
  --ref-video-audio                        WAV soundtrack paired by index with --ref-video (can be used multiple times)
  --ref-audio                              standalone WAV reference for MiniMax-H3 Ref2VA (can be used multiple times)
  --audio                                  driving audio track (Wan2.2 S2V; can be used once)
  --cache-mode                             caching method: 'easycache' (DiT), 'ucache' (UNET),
                                           'dbcache'/'taylorseer'/'cache-dit' (DiT block-level), 'spectrum' (UNET/DiT
                                           Chebyshev+Taylor forecasting)
  --cache-option                           named cache params (key=value format, comma-separated). easycache/ucache:
                                           threshold=,start=,end=,decay=,relative=,reset=; dbcache/taylorseer/cache-dit:
                                           Fn=,Bn=,threshold=,warmup=; spectrum: w=,m=,lam=,window=,flex=,warmup=,stop=.
                                           Examples: "threshold=0.25" or "threshold=1.5,reset=0"
  --scm-mask                               SCM steps mask for cache-dit: comma-separated 0/1 (e.g.,
                                           "1,1,1,0,0,1,0,0,1,0") - 1=compute, 0=can cache
  --scm-policy                             SCM policy: 'dynamic' (default) or 'static'
  --vae-tile-size                          tile size for vae encode and decode in image pixels, format [W]x[H] or [S]
                                           (default: 256x256; requires --vae-tiling)
  --vae-relative-tile-size                 relative tile size for vae encode and decode, format [W]x[H] or [S]: <=1 is a
                                           dimension fraction, >1 a target tile count (overrides --vae-tile-size;
                                           requires --vae-tiling)
  --prompt-file                            path to the file containing the prompt to render
  --negative-prompt-file                   path to the file containing the negative prompt
```
