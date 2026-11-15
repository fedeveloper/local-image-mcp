# stable-diffusion.cpp backend (`sdcpp`)

The `sdcpp` backend runs the stable-diffusion.cpp command line program `sd-cli` once per
image. It is the default backend for the built-in models.

**Verified version:** stable-diffusion.cpp commit `532d2c1` (2026-11-15).
`local-image doctor` shows the version of your `sd-cli`. If the commit differs, the
backend still runs, but `doctor` shows a note, because command line options can change.

> **Note:** older builds named the program `sd` (or `sd.exe`). Current builds name it
> `sd-cli`. The backend looks for `sd-cli` first, then `sd`.

## 1. Install sd-cli

### Windows 11 (NVIDIA GPU)

1. Open the [releases page](https://github.com/leejet/stable-diffusion.cpp/releases).
2. Download the Windows archive for CUDA 12 (x64).
3. Extract the archive to a folder, for example `C:\tools\stable-diffusion.cpp`.
4. Open PowerShell in that folder and run `.\sd-cli.exe --version`.

> **Note for RTX 50 series (for example RTX 5060):** the CUDA 12 release build
> (commit `5dcfa79`) works on an RTX 5060. If your build does not use the GPU, or the
> first run is very slow, build it from source (below) with CUDA 12.8 or later and
> `-DCMAKE_CUDA_ARCHITECTURES=120`.

Build from source (needs Visual Studio 2022, CMake and the CUDA Toolkit 12.8+):

```powershell
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
cmake -B build -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release
.\build\bin\Release\sd-cli.exe --version
```

### Linux

Use a release archive, or build from source with CUDA (or Vulkan with `-DSD_VULKAN=ON`):

```sh
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
cmake -B build -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
./build/bin/sd-cli --version
```

The official build guide is `docs/build.md` in the stable-diffusion.cpp repository.

## 2. Tell local-image-mcp where sd-cli is

Put `sd-cli` on `PATH`, or set `LOCAL_IMAGE_SD_BINARY` to its full path.

Windows (PowerShell, for the current user):

```powershell
[Environment]::SetEnvironmentVariable("LOCAL_IMAGE_SD_BINARY", "C:\tools\stable-diffusion.cpp\sd-cli.exe", "User")
```

Linux:

```sh
export LOCAL_IMAGE_SD_BINARY="$HOME/stable-diffusion.cpp/build/bin/sd-cli"
```

Then check the setup:

```sh
local-image doctor
local-image models check
```

## 3. Generate a test image

```sh
local-image generate --model z-image-turbo --seed 42 --prompt "A red fox sitting in fresh snow at sunrise, soft warm light, shallow depth of field, photorealistic"
```

## What the backend does

For each image, the backend:

1. writes the prompt to a temporary UTF-8 file and passes it with `--prompt-file`
   (no shell quoting problems with quotes, `%`, new lines or non-ASCII text);
2. passes the model files from the registry (`--diffusion-model`, `--llm`, `--vae`),
   the size, steps, `--cfg-scale`, `--seed`, sampler and reference images (`-r`);
3. reads the console output for progress (loading, encoding, sampling steps, decoding);
4. writes to a temporary file name, then renames it to the final name;
5. reads the generation record that `sd-cli` embeds in the PNG
   (`sd-cli -M metadata ... --metadata-format json`) and copies it into the sidecar JSON.

Cancel terminates the `sd-cli` process, so the GPU memory is free at once.

## Reference images (FLUX.2 klein)

`flux2-klein-4b` has the capabilities `reference` and `edit`. Give up to 4 images
(`--ref` in the CLI, `reference_images` in the MCP tool). The backend passes each one
with `-r`. sd-cli selects the `flux2` reference preset by itself (VAE-encoded references
go to the DiT, reference index `increase`; see `docs/guides/edit.md` at commit
`532d2c1`), so no extra option is necessary.

- Formats: PNG, JPEG, WebP, BMP.
- Relative paths start at the project directory.
- Without `width` and `height`, the output keeps the aspect ratio of the first reference,
  at about the default pixel count of the model, inside the model limits.
- The sidecar JSON records the path and the SHA-256 of each reference.

Example:

```sh
uv run local-image generate -m flux2-klein-4b --seed 42 --ref photo.jpg \
  -p "Make the jacket bright blue. Keep the face, the pose and the background."
```

## Server mode (sd-server)

In CLI mode, the backend starts `sd-cli` for each image, so each image loads the model
again. In server mode, the backend runs `sd-server` (in the same release archive as
`sd-cli`) and keeps the model loaded between images. On the test PC, the second and the
next images with the same model take about 14 s instead of 22 s.

`LOCAL_IMAGE_SD_MODE` selects the mode:

- `auto` (default): server mode when `sd-server` is found, else CLI mode;
- `cli`: always `sd-cli`;
- `server`: always `sd-server` (the backend is not available without it).

- The first job starts `sd-server` with the model files of the job. It listens on
  `127.0.0.1` only, on a free port.
- The next jobs with the same model and the same options use the loaded model.
- A job with a different model (or different options) restarts `sd-server`.
- When no job waits, `sd-server` stops after a grace period (default 120 s), and the
  GPU memory is free again. The grace period exists because Claude sends jobs one after
  the other, and the queue is empty for a short time between two jobs.
- Cancel sends a cancel request to `sd-server`. If the job does not stop within 30 s,
  the backend terminates `sd-server`.

The backend uses the native asynchronous API of `sd-server` (`/sdcpp/v1/img_gen`,
`/sdcpp/v1/jobs/{id}`, `/sdcpp/v1/jobs/{id}/cancel`, `/sdcpp/v1/models/status`),
verified at commit `532d2c1`.

To use it:

1. Make sure that `sd-server` (`sd-server.exe` on Windows) is next to `sd-cli`, or set
   `LOCAL_IMAGE_SD_SERVER_BINARY` to its full path.
2. Optional: set `LOCAL_IMAGE_SD_SERVER_IDLE` to the grace period in seconds
   (`0` stops `sd-server` as soon as the queue is empty).
3. Run `uv run local-image doctor`. It shows `sd mode: auto (server)` and the
   `sd-server` path.

To go back to one `sd-cli` process per image, set `LOCAL_IMAGE_SD_MODE=cli` (for Claude
Code, add `-e LOCAL_IMAGE_SD_MODE=cli` to `claude mcp add`).

NOTE: While `sd-server` waits in the grace period, it keeps its GPU memory. Another
`local-image-mcp` process (for example a second Claude Code session) can start a job in
that time and run out of memory. Use `LOCAL_IMAGE_SD_SERVER_IDLE=0` if you often run
several sessions at the same time.

The output of `sd-server` goes to `<state dir>/logs/sd-server.log`. The sidecar JSON has
`backend.options.mode = "server"` and the generation record of `sd-server`.

## Backend options (registry `backend_options`)

| Option | Default | Effect |
|---|---|---|
| `offload_to_cpu` | `false` (built-in models: `true`) | `--offload-to-cpu`: keep weights in RAM, load them into VRAM when needed. |
| `vae_tiling` | `false` | `--vae-tiling`: decode in tiles to save VRAM. |
| `text_encoder_on_cpu` | `false` (built-in models: `true`) | `--backend te=cpu`: run the text encoder on the CPU. Needed for the ~8 GB diffusers text encoders on an 8 GB GPU. |
| `flash_attn` | not set | `--flash-attn on\|off\|auto` (sd-cli default: `auto`). |
| `timeout_s` | `1800` | Stop the job after this many seconds. |
| `server_start_timeout_s` | `600` | Server mode: maximum time for `sd-server` to load the model. |
| `extra_args` | `[]` | Extra `sd-cli` arguments, passed as they are (in server mode, they go to `sd-server`). |

## Troubleshooting

- **`sd-cli not found`**: set `LOCAL_IMAGE_SD_BINARY` to the full path of `sd-cli.exe`.
- **Out of memory**: keep `offload_to_cpu: true`, add `vae_tiling: true`, or use a smaller
  size (for example 768×768).
- **`uv sync` fails on Windows with "file in use" (`local-image-mcp.exe`)**: a Claude Code
  session runs the `local-image` MCP server and locks the file. Close the Claude Code
  sessions, then run `uv sync` again.
- **Black or white images**: see `docs/troubleshooting.md` in the stable-diffusion.cpp
  repository.
- **The job fails**: the error message contains the `[E]` lines of `sd-cli`. The full
  raw output of each run is in `<state dir>/logs/<image name>.sd-cli.log`
  (`local-image doctor` shows the state dir). The sidecar JSON has the same path in
  `backend.log_file`, and the exact `argv`.

## Tested setup

| Date | OS / GPU | sd-cli | Model files | Result |
|---|---|---|---|---|
| 2026-11-15 | Windows 11, RTX 5060 8 GB, CUDA 12 release build | commit `5dcfa79` | Z-Image-Turbo Q5_K_M GGUF, diffusers text encoder (on CPU) and VAE from `unsloth/Z-Image-Turbo` | 1024×1024, 8 steps: 35.7 s cold, 22.4 s warm; peak VRAM about 5.3 GB; same seed gives a byte-identical PNG |
| 2026-11-15 | same | same | Z-Image-Turbo Q5_K_M with the built-in registry (no user override) | 1024×1024, 8 steps: 35.2 s; peak VRAM about 5.3 GB above the desktop baseline; progress reaches 8/8 |
| 2026-11-15 | same | same | FLUX.2-klein-4B Q4_K_M GGUF, diffusers text encoder (on CPU) and VAE from `unsloth/FLUX.2-klein-4B` | 1024×1024, 4 steps: 21.9 s; peak VRAM about 3.6 GB above the desktop baseline; progress reaches 4/4 |
| 2026-11-15 | same | same | FLUX.2-klein-4B Q4_K_M, one reference image (1216×832) | edit "make the jacket blue": output 1232×848 from the reference shape, 4 steps: 25.9 s; peak VRAM about 4.6 GB above the desktop baseline; pose and background kept |
| 2026-11-15 | same | same, `sd-server.exe` from the same release | Server mode: 3× Z-Image-Turbo, then FLUX.2 klein 4B, then Z-Image-Turbo (1024×1024, seed 42) | Z-Image: 21.8 s with model load, then 13.9 s and 14.0 s with the loaded model; model change restarts sd-server (22.2 s, 21.5 s); peak VRAM 6.3 GB; all VRAM free after the queue stops; same PNG (SHA-256) as CLI mode |
