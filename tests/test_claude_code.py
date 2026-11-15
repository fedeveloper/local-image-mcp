"""Claude Code integration files and the `local-image claude` commands."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

from local_image_mcp import claude_code, cli
from local_image_mcp.claude_code import SkillScope
from local_image_mcp.errors import LocalImageError
from local_image_mcp.registry import load_registry


def _frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n")
    head = text.split("---\n", 2)[1]
    data: object = yaml.safe_load(head)
    assert isinstance(data, dict)
    items = cast("dict[object, object]", data).items()
    return {str(k): str(v) for k, v in items}


def test_skill_frontmatter() -> None:
    meta = _frontmatter(claude_code.skill_text())
    assert meta["name"] == "local-image"
    assert 20 < len(meta["description"]) <= 1024


def test_skill_names_only_real_tools_and_models() -> None:
    text = claude_code.skill_text()
    for tool in ("list_models", "generate_image", "get_job_status", "get_job_result"):
        assert f"`{tool}`" in text
    models = set(load_registry().models)
    for model_id in ("z-image-turbo", "flux2-klein-4b"):
        assert model_id in models
        assert f"`{model_id}`" in text


def test_skill_sizes_are_multiples_of_16() -> None:
    import re

    for w, h in re.findall(r"(\d{3,4})\u00d7(\d{3,4})", claude_code.skill_text()):
        assert int(w) % 16 == 0 and int(h) % 16 == 0


def test_skill_path_scopes(tmp_path: Path) -> None:
    home, project = tmp_path / "home", tmp_path / "project"
    assert claude_code.skill_path(SkillScope.USER, project, home) == (
        home / ".claude" / "skills" / "local-image" / "SKILL.md"
    )
    assert claude_code.skill_path(SkillScope.PROJECT, project, home) == (
        project / ".claude" / "skills" / "local-image" / "SKILL.md"
    )


def test_install_skill_is_idempotent_and_protects_changes(tmp_path: Path) -> None:
    target = tmp_path / "skills" / "local-image" / "SKILL.md"
    assert claude_code.install_skill(target) is True
    assert claude_code.install_skill(target) is False
    target.write_text("my own changes", encoding="utf-8")
    with pytest.raises(LocalImageError, match="--force"):
        claude_code.install_skill(target)
    assert claude_code.install_skill(target, force=True) is True
    assert target.read_text(encoding="utf-8") == claude_code.skill_text()


def test_cli_install_skill_project_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("LOCAL_IMAGE_PROJECT_DIR", str(project))
    monkeypatch.setenv("LOCAL_IMAGE_STATE_DIR", str(tmp_path / "state"))
    assert cli.main(["claude", "install-skill", "--scope", "project"]) == cli.EXIT_OK
    assert (project / ".claude" / "skills" / "local-image" / "SKILL.md").is_file()
    assert "Installed the skill" in capsys.readouterr().out
    assert cli.main(["claude", "install-skill", "--scope", "project"]) == cli.EXIT_OK
    assert "up to date" in capsys.readouterr().out


def test_cli_prints_files(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["claude", "show-skill"]) == cli.EXIT_OK
    assert capsys.readouterr().out == claude_code.skill_text()
    assert cli.main(["claude", "claude-md"]) == cli.EXIT_OK
    assert capsys.readouterr().out.startswith("## Images (local-image MCP)")
