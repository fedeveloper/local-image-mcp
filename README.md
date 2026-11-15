# local-image-mcp

Generate images on your own GPU from Claude Code.

`local-image-mcp` is an [MCP](https://modelcontextprotocol.io) server and a command line
tool. Claude does the reasoning: it writes the prompt and decides what the project needs.
The pixels come from open image models that run locally, from GGUF files in your
Hugging Face cache, through [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

- Two ready models: **Z-Image-Turbo** (text to image) and **FLUX.2 klein 4B**
  (text to image, and edits with up to 4 reference images).
- Runs on an 8 GB GPU. Tested on Windows 11 with an RTX 5060 (8 GB): about 22–36 s
  for a 1024×1024 image.
- Every PNG has a JSON sidecar with all settings, so you can generate it again.
- No cloud service. The server never downloads anything; only
  `local-image models pull` does, and it reuses the files that are already in your cache.

> **Status: release candidate (0.1.0rc1).** Windows is the main platform; Linux works
> and runs in CI. See [CHANGELOG.md](CHANGELOG.md) and [PLAN.md](PLAN.md).

## How it works

```
Claude Code ──MCP (stdio)──▶ local-image-mcp ──▶ job queue (one job at a time)
                                   │                      │
                       model registry (YAML)        sd-cli (GPU)
                                   │                      │
                       Hugging Face cache ◀───────────────┘
                                                   PNG + JSON sidecar
```

The server has five tools: `list_models`, `generate_image`, `get_job_status`,
`get_job_result` and `cancel_job`. Each tool returns in less than a minute, so no
timeout setting is necessary: the generation runs in a background job.

## Requirements

- Windows 11 or Linux, with an NVIDIA GPU (8 GB VRAM or more) and a recent driver.
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).
- `sd-cli` from stable-diffusion.cpp, CUDA build.
- About 25 GB of disk space for the two models (less with GGUF text encoders).
- [Claude Code](https://code.claude.com), for the MCP integration.

## Install on Windows 11

1. Install uv (PowerShell):
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Download the stable-diffusion.cpp release for Windows, CUDA 12, x64, from the
   [releases page](https://github.com/leejet/stable-diffusion.cpp/releases).
   Extract it, for example to `C:\tools\sdcpp\`.
3. Clone this repository and install it:
   ```powershell
   git clone https://github.com/fedeveloper/local-image-mcp
   cd local-image-mcp
   uv sync
   ```
4. Tell the tool where `sd-cli.exe` is (for your user):
   ```powershell
   [Environment]::SetEnvironmentVariable("LOCAL_IMAGE_SD_BINARY", "C:\tools\sdcpp\sd-cli.exe", "User")
   ```
   Open a new terminal after this step.
5. Check the setup: `uv run local-image doctor`.

## Install on Linux

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Build `sd-cli` with CUDA (or use a release archive):
   ```sh
   git clone --recursive https://github.com/leejet/stable-diffusion.cpp
   cd stable-diffusion.cpp
   cmake -B build -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -j
   ```
3. Clone this repository and install it:
   ```sh
   git clone https://github.com/fedeveloper/local-image-mcp
   cd local-image-mcp
   uv sync
   ```
4. Tell the tool where `sd-cli` is, for example in `~/.bashrc`:
   `export LOCAL_IMAGE_SD_BINARY=$HOME/stable-diffusion.cpp/build/bin/sd-cli`
5. Check the setup: `uv run local-image doctor`.

More details, build options and troubleshooting: [docs/backends/sdcpp.md](docs/backends/sdcpp.md).

For several images in a row, enable the server mode (`LOCAL_IMAGE_SD_MODE=server`): the
model stays loaded in `sd-server` between images. See
[server mode](docs/backends/sdcpp.md#server-mode-sd-server).

## Get the models

1. See which files are already in your Hugging Face cache:
   `uv run local-image models check`
2. Download only the missing files of a model:
   `uv run local-image models pull z-image-turbo`
3. Generate a test image:
   ```sh
   uv run local-image generate -m z-image-turbo --seed 42 -p "A red fox in fresh snow at sunrise, soft warm light, photorealistic"
   ```

The tool respects `HF_HOME` and `HF_HUB_CACHE`. Each model accepts several quantizations
(Q4_K_M, Q5_K_M, Q6_K, ...) and text encoders in GGUF or diffusers format: the first
file found in your cache is used, so you do not download a second copy.

## Use it from Claude Code

1. Register the server for your user. Windows (PowerShell; use `claude.exe`, and put the
   name before `-e`):
   ```powershell
   claude.exe mcp add --transport stdio --scope user local-image `
     -e LOCAL_IMAGE_SD_BINARY=C:\tools\sdcpp\sd-cli.exe `
     -- uv run --directory C:\path\to\local-image-mcp local-image-mcp
   ```
   Linux:
   ```sh
   claude mcp add --transport stdio --scope user local-image \
     -e LOCAL_IMAGE_SD_BINARY=$HOME/stable-diffusion.cpp/build/bin/sd-cli \
     -- uv run --directory /path/to/local-image-mcp local-image-mcp
   ```
2. Install the skill, so that Claude knows the workflow and how to write prompts:
   `uv run local-image claude install-skill`
3. Start a new Claude Code session and run `/mcp`: `local-image` must be connected.
4. Ask for an image, for example: *"Create a 1216×832 hero image of a mountain cabin at
   dusk and save it in public/images/hero.png."*

Images go to `<project>/generated-images/` unless Claude gives an `output_path`.
`uv run local-image claude claude-md` prints a section for your project's `CLAUDE.md`
(asset folder, sizes, style). Full guide: [docs/claude-code.md](docs/claude-code.md).

## Models

| Model id | Model | Capabilities | Steps | VRAM (measured, 1024²) | License of the weights |
|---|---|---|---|---|---|
| `z-image-turbo` | Z-Image-Turbo (GGUF) | text2img | 8 | ~5.3 GB | Apache-2.0 |
| `flux2-klein-4b` | FLUX.2 klein 4B (GGUF) | text2img, reference, edit | 4 | ~3.6 GB; ~4.6 GB with a reference | Apache-2.0 |

Measured on an RTX 5060 8 GB with the text encoder on the CPU. You can add or change
models with your own registry file (`LOCAL_IMAGE_REGISTRY`); see the built-in
[`models.yaml`](src/local_image_mcp/registry/models.yaml).

### Model licenses

**The license of this code (Apache-2.0) does not cover the models.** Each model has its
own license, shown by `local-image models list` and by the `list_models` tool. Read it
before you use the images, above all for commercial use. For example, FLUX.2 klein
**9B** and FLUX.2 dev use a **non-commercial** license; they are not in the built-in
registry for that reason.

## Command line

| Command | What it does |
|---|---|
| `local-image doctor` | Shows the configuration, the `sd-cli` version and the cache. |
| `local-image models list / check / pull` | Lists models, checks the cache, downloads missing files. |
| `local-image generate -m <model> -p "<prompt>"` | Generates one image (`--seed`, `-W/-H`, `--steps`, `--ref`, `-o`). |
| `local-image regenerate <image>.json` | Generates an image again from its sidecar. |
| `local-image claude install-skill / claude-md` | Installs the Claude Code skill, prints the CLAUDE.md section. |

## Responsible use

- **Do not generate images of real, identifiable people without their consent.** This
  includes public figures, colleagues, friends and family.
- Do not create images that deceive (fake evidence, fake news photos, fake documents),
  that harass, or that sexualize anyone. Never create sexual content that involves minors.
- Label generated images as AI-generated when people could take them for real photos.
- Respect copyright and trademarks in prompts and in reference images.
- Follow the license and the use policy of each model.

The skill for Claude Code contains the same rules.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

The tests use a `fake` backend and need no GPU. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
