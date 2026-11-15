# Models, quantizations and your own registry

The tool knows a model only from its **registry**: the built-in
[`models.yaml`](../src/local_image_mcp/registry/models.yaml), plus an optional user
registry. The server never downloads files and never guesses file names. This page tells
you what to do when your files are not the ones in the built-in registry.

## Which case is yours?

| Your situation | What to do |
|---|---|
| You have another quantization of a built-in model that is in the list (Q4_K_M, Q4_K_S, Q5_K_M, Q6_K, Q8_0, Q3_K_M for Z-Image). | Nothing. The first file found in your cache is used. |
| You have several quantizations in the cache and want one of them. | Set `LOCAL_IMAGE_QUANT`, for example `Q8_0`. |
| You want to download a quantization other than the default (Q4_K_M). | `local-image models pull <model> --quant Q6_K`, then set `LOCAL_IMAGE_QUANT=Q6_K`. |
| Your file is not in the list (another quantization, another repository, a local file). | Write a user registry. |
| You want another model. | Write a user registry. stable-diffusion.cpp must support the model architecture. |

Run `local-image models check` to see which file the tool uses for each model.

## How the tool picks a file

Each file of a model (diffusion model, text encoder, VAE) has a main entry and a list of
`alternatives`. The tool uses the first one that is in your Hugging Face cache
(`HF_HUB_CACHE`, else `HF_HOME/hub`) or on disk (`path`).

`LOCAL_IMAGE_QUANT` moves the entries whose file name contains that quantization to the
front. The match uses the whole token and ignores case: `Q8_0` matches
`z-image-turbo-Q8_0.gguf`, and `Q4_K` does not match `Q4_K_M`. If no entry of a file
matches, or the matching file is not in the cache, the normal order applies.

For Claude Code, pass it to the MCP server:
`claude mcp add ... -e LOCAL_IMAGE_QUANT=Q8_0 ...`.

## Write a user registry

1. Create the file with comments and an example:
   `uv run local-image models init-registry`
   The file goes to `LOCAL_IMAGE_REGISTRY` if you set it. Otherwise it goes to the user
   config folder: `%LOCALAPPDATA%\local-image-mcp\models.yaml` on Windows,
   `~/.config/local-image-mcp/models.yaml` on Linux. `local-image doctor` shows the path.
2. Add your models under `models:`. The rules:
   - a new id adds a model;
   - the id of a built-in model replaces the **whole** built-in entry (copy all of it);
   - `<id>: null` hides a built-in model.
3. Check it: `uv run local-image models check`.
4. Test it: `uv run local-image generate -m <id> -p "..."`.
5. Start a new Claude Code session. The MCP server reads the registry only when it starts.

NOTE: Each person can keep a user registry. It is not in the repository, so your files do
not change the setup of other people.

### Example: your own files for Z-Image-Turbo

```yaml
schema_version: 1
models:
  my-z-image:
    display_name: Z-Image-Turbo (my files)
    backend: sdcpp
    capabilities: [text2img]
    license: apache-2.0
    files:
      diffusion_model:
        path: D:/models/z-image-turbo-Q3_K_S.gguf      # a local file
      text_encoder:
        repo: unsloth/Qwen3-4B-Instruct-2507-GGUF       # a file in the HF cache
        filename: Qwen3-4B-Instruct-2507-Q8_0.gguf
      vae:
        repo: Comfy-Org/z_image_turbo
        filename: split_files/vae/ae.safetensors
    defaults: {width: 1024, height: 1024, steps: 8, cfg_scale: 1.0}
    limits: {min_size: 256, max_size: 2048, size_multiple: 16, max_steps: 16}
    backend_options: {offload_to_cpu: true, text_encoder_on_cpu: true}
```

The file roles map to sd.cpp options: `diffusion_model` → `--diffusion-model`,
`text_encoder` → `--llm`, `vae` → `--vae`, `model` → `--model` (a single all-in-one
file). See [backends/sdcpp.md](backends/sdcpp.md).

### Fields

| Field | Meaning |
|---|---|
| `display_name`, `license`, `license_url` | Shown by `models list` and the `list_models` tool. |
| `backend` | `sdcpp`. |
| `capabilities` | `text2img`; add `reference` and `edit` with `max_reference_images` (1–4) for models that take reference images. |
| `files.<role>` | `repo` + `filename` (+ optional `revision`), or `path`. Optional `alternatives`: a list of the same. |
| `defaults` | `width`, `height`, `steps`, `cfg_scale`, optional `sampling_method`, `scheduler`. |
| `limits` | `min_size`, `max_size`, `size_multiple`, `max_pixels`, `min_steps`, `max_steps`. |
| `backend_options` | See [backends/sdcpp.md](backends/sdcpp.md#backend-options-registry-backend_options). |
| `prompt_notes`, `supports_negative_prompt`, `vram_estimate_gb` | Hints for Claude, shown by `list_models`. |

The registry is strict: an unknown field or a default outside the limits is an error
that names the field.
