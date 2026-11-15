# local-image-mcp

> **Status: early development (Phase 2 of 7).** The core, the model registry, the CLI
> and the stable-diffusion.cpp backend work. The MCP server (Phase 3) is not ready yet.
> See [PLAN.md](PLAN.md).

`local-image-mcp` connects Claude Code (and other MCP clients) to image generation models
that run on your own GPU. Claude does the reasoning. When a project needs an image, it
calls this server, and the image is generated locally from GGUF models in your
Hugging Face cache.

## Development quick start

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run local-image models list
uv run local-image generate --model fake --prompt "a red fox in the snow" --seed 42
uv run local-image doctor
```

The `fake` model writes a solid-colour PNG without a GPU. Every image gets a JSON file
with the same name. The JSON contains all settings, so you can generate the image again:

```sh
uv run local-image regenerate generated-images/<name>.json
```

Checks (the same as CI):

```sh
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

## Generate with a real model

Install `sd-cli` (stable-diffusion.cpp) as described in
[docs/backends/sdcpp.md](docs/backends/sdcpp.md), then:

```sh
uv run local-image doctor
uv run local-image models check z-image-turbo
uv run local-image generate --model z-image-turbo --seed 42 --prompt "A red fox in fresh snow at sunrise, soft warm light, photorealistic"
```

## Models

| Model | Capabilities | VRAM (estimate) | License of the weights |
|---|---|---|---|
| `z-image-turbo` (Z-Image-Turbo Q4_K_M) | text2img | ~7 GB | Apache-2.0 |
| `flux2-klein-4b` (FLUX.2 klein 4B Q4_K_M) | text2img, reference, edit | ~6 GB | Apache-2.0 |

**The code license (Apache-2.0) does not cover the models.** Check the license of each model
before you use its output. For example, FLUX.2 klein **9B** uses a non-commercial license.

`local-image models check` shows which model files are in your Hugging Face cache
(`HF_HUB_CACHE` / `HF_HOME` are respected). Only `local-image models pull <model>`
downloads files, and it skips files that are already in the cache.

## License

Apache-2.0. See [LICENSE](LICENSE).
