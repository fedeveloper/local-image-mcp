# Releasing

ATTENTION: A version on PyPI cannot be replaced. Check every step before you push a tag.

## One-time setup

1. Create the project on PyPI as a "pending publisher":
   PyPI → Your account → Publishing → Add a new pending publisher.
2. Fill in: project name `local-image-mcp`, owner `fedeveloper`, repository
   `local-image-mcp`, workflow `release.yml`, environment `pypi`.
3. On GitHub, create the environment `pypi` (Settings → Environments). Optionally add
   yourself as a required reviewer, so that each publish waits for your approval.

## Each release

1. Set the version in `pyproject.toml` and in `src/local_image_mcp/__init__.py`
   (a test checks that they match).
2. Move the changes in `CHANGELOG.md` under the new version, with the date.
3. Merge the pull request into `main` and wait for green CI.
4. Tag the merge commit and push the tag:
   ```sh
   git tag v0.1.0rc1
   git push origin v0.1.0rc1
   ```
5. The `Release` workflow checks the tag, runs the tests, builds and publishes.
6. Create a GitHub release from the tag and paste the changelog section.

After the first release, the Claude Code command can use `uvx`:

```sh
claude mcp add --transport stdio --scope user local-image -- uvx local-image-mcp
```
