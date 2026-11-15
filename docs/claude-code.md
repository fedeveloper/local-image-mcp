# Use local-image-mcp with Claude Code

The MCP server runs over stdio. Claude Code starts it and calls its five tools:

| Tool | What it does | Returns in |
|---|---|---|
| `list_models` | Lists the models, their capabilities, defaults, limits and availability. | < 1 s |
| `generate_image` | Validates the request and queues one job. Returns the `job_id`. | < 1 s |
| `get_job_status` | Returns the state and progress. With `wait_seconds` (0–45) it waits for the end. | ≤ 45 s |
| `get_job_result` | Returns the image path, the metadata path and the settings. `include_preview=true` adds a small JPEG (about 384 px). | < 2 s |
| `cancel_job` | Cancels a queued or running job. | < 5 s |

The server runs one job at a time. Images go to `<project>/generated-images/` by default,
with a JSON sidecar that can reproduce them (`local-image regenerate <file>.json`).

## Register the server

Before the first PyPI release, run the server from a clone of this repository.

1. Clone the repository and run `uv sync` in it.
2. Install `sd-cli` and check the models (see [backends/sdcpp.md](backends/sdcpp.md)):
   `uv run local-image doctor` and `uv run local-image models check`.
3. Register the server for your user.

Windows (PowerShell):

```powershell
claude mcp add --transport stdio --scope user `
  -e LOCAL_IMAGE_SD_BINARY=C:\tools\sdcpp\sd-cli.exe `
  local-image -- uv run --directory C:\path\to\local-image-mcp local-image-mcp
```

Linux:

```sh
claude mcp add --transport stdio --scope user \
  -e LOCAL_IMAGE_SD_BINARY=$HOME/sdcpp/build/bin/sd-cli \
  local-image -- uv run --directory /path/to/local-image-mcp local-image-mcp
```

After the release, the command becomes `... local-image -- uvx local-image-mcp`.

4. Start Claude Code in a project and run `/mcp`. The `local-image` server must show as
   connected.

NOTE: Set `HF_HOME` or `HF_HUB_CACHE` with `-e` if your Hugging Face cache is not in the
default location and these variables are not set for your user.

## Environment variables

| Variable | Meaning |
|---|---|
| `LOCAL_IMAGE_SD_BINARY` | Path of `sd-cli` (if it is not on `PATH`). |
| `HF_HUB_CACHE` / `HF_HOME` | Hugging Face cache location. |
| `LOCAL_IMAGE_OUTPUT_DIR` | Default output directory. Default: `<project>/generated-images`. |
| `LOCAL_IMAGE_ALLOWED_ROOTS` | Extra directories where output files are allowed (`;` on Windows, `:` on Linux). |
| `LOCAL_IMAGE_REGISTRY` | A user registry YAML file that adds or replaces models. |
| `LOCAL_IMAGE_STATE_DIR` | Jobs, GPU lock and sd-cli logs. Default: `%LOCALAPPDATA%\local-image-mcp` on Windows. |
| `LOCAL_IMAGE_LOG_LEVEL` | Server log level on stderr (default `WARNING`). |
| `LOCAL_IMAGE_ENABLE_FAKE` | `1` shows the `fake` test model. For development only. |

Claude Code sets `CLAUDE_PROJECT_DIR`. The server uses it as the project directory, so
relative output paths and reference images resolve against your project.

## Timeouts

No timeout setting is necessary. Every tool returns in less than 60 seconds, and the
image generation runs in a background job. `get_job_status` sends progress notifications
while it waits.

- If the first start of the server is slow (for example, the first `uvx` download), set
  `MCP_TIMEOUT=30000` (milliseconds) before you start Claude Code.
- `MCP_TOOL_TIMEOUT` and `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT` do not affect this server,
  because no tool call is long.

## Output size

The tools return paths and short JSON, not image bytes. The optional preview is a small
JPEG, well below the 10,000-token warning of Claude Code.
