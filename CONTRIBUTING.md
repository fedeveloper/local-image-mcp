# Contributing

Thank you for your help. This project is small, so the rules are short.

## Set up

1. Install [uv](https://docs.astral.sh/uv/).
2. Clone the repository and run `uv sync`.
3. Run the checks (the same as CI):
   ```sh
   uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
   ```

The tests use the `fake` backend and `tests/fake_sd_cli.py`. They need no GPU and no
model files. CI runs them on Ubuntu and Windows.

## Rules

- Write code, comments and docs in English. Use short, simple sentences in the docs.
- Keep `pyright` strict and `ruff` clean. Add tests for new behaviour.
- **Do not guess command line options or file names.** For a change to the `sdcpp`
  backend, read `sd-cli --help` and the stable-diffusion.cpp source of the version you
  test, and write the verified commit in the code or the docs.
- **Do not guess model file names.** For a new registry entry, list the files of the
  Hugging Face repository and copy the exact names. Write the license of each file.
- Do not add models with an unclear license to the built-in registry.
- The MCP server must never download files. Only `local-image models pull` downloads.
- Every tool of the MCP server must return in less than 60 seconds.

## Add a model

1. Add an entry to `src/local_image_mcp/registry/models.yaml` (or test it first in your
   own registry file with `LOCAL_IMAGE_REGISTRY`).
2. Run `local-image models check <id>` and generate a few images.
3. Write the measured time and VRAM in the comment and in `docs/backends/sdcpp.md`.
4. Add the model to the table in `README.md` and, if it fits, to the skill
   (`src/local_image_mcp/claude_code/SKILL.md`).

## Pull requests

- One topic per pull request.
- Describe what you tested, and on which GPU and operating system.
- For a real-GPU change, add a line to the "Tested setup" table in
  `docs/backends/sdcpp.md`.
