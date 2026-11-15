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

> **Note for RTX 50 series (for example RTX 5060):** these GPUs need a build with
> CUDA 12.8 or later. If `sd-cli` does not use the GPU, or the first run is very slow,
> build it from source (below) with `-DCMAKE_CUDA_ARCHITECTURES=120`.

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

## Backend options (registry `backend_options`)

| Option | Default | Effect |
|---|---|---|
| `offload_to_cpu` | `false` (built-in models: `true`) | `--offload-to-cpu`: keep weights in RAM, load them into VRAM when needed. |
| `vae_tiling` | `false` | `--vae-tiling`: decode in tiles to save VRAM. |
| `text_encoder_on_cpu` | `false` (built-in models: `true`) | `--backend te=cpu`: run the text encoder on the CPU. Needed for the ~8 GB diffusers text encoders on an 8 GB GPU. |
| `flash_attn` | not set | `--flash-attn on\|off\|auto` (sd-cli default: `auto`). |
| `timeout_s` | `1800` | Stop the job after this many seconds. |
| `extra_args` | `[]` | Extra `sd-cli` arguments, passed as they are. |

## Troubleshooting

- **`sd-cli not found`**: set `LOCAL_IMAGE_SD_BINARY` to the full path of `sd-cli.exe`.
- **Out of memory**: keep `offload_to_cpu: true`, add `vae_tiling: true`, or use a smaller
  size (for example 768×768).
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
