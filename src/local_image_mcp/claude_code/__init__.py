"""Claude Code integration files: the ``local-image`` skill and a CLAUDE.md section.

Claude Code loads skills from ``~/.claude/skills/<name>/SKILL.md`` (user) and from
``<project>/.claude/skills/<name>/SKILL.md`` (project).
"""

from __future__ import annotations

from enum import StrEnum
from importlib import resources
from pathlib import Path

from local_image_mcp.errors import LocalImageError

SKILL_NAME = "local-image"
_SKILL_FILE = "SKILL.md"
_CLAUDE_MD_FILE = "claude_md_section.md"


class SkillScope(StrEnum):
    USER = "user"
    PROJECT = "project"


def skill_text() -> str:
    return resources.files(__package__).joinpath(_SKILL_FILE).read_text("utf-8")


def claude_md_section() -> str:
    return resources.files(__package__).joinpath(_CLAUDE_MD_FILE).read_text("utf-8")


def skill_path(scope: SkillScope, project_dir: Path, home: Path | None = None) -> Path:
    base = (home or Path.home()) if scope is SkillScope.USER else project_dir
    return base / ".claude" / "skills" / SKILL_NAME / _SKILL_FILE


def install_skill(target: Path, *, force: bool = False) -> bool:
    """Write the skill to ``target``. Return ``False`` if it is already up to date.

    Raises ``LocalImageError`` if a different file exists and ``force`` is false.
    """
    text = skill_text()
    if target.exists():
        if target.read_text("utf-8") == text:
            return False
        if not force:
            raise LocalImageError(f"{target} exists and is different. Use --force to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return True
