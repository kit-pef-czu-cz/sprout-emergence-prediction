"""Shared helpers for resolving project-scoped pipeline paths from TOML config."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH: Path = Path(__file__).resolve().parents[1] / "config" / "paths.toml"


@dataclass(frozen=True)
class ConfigContext:
    """Shared context loaded from the top-level [paths] TOML table."""

    config_path: Path
    project_root: Path
    project_name: str


def get_required_string(config: dict[str, Any], key: str, config_path: Path) -> str:
    """Return a required non-empty string value from a TOML mapping."""
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{key}' to be a non-empty string in {config_path}.")
    return value


def resolve_project_path(
    project_root: Path, relative_path: str, key: str, config_path: Path
) -> Path:
    """Resolve a relative config path against the configured project root."""
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        raise ValueError(f"Expected '{key}' in {config_path} to be a relative path.")
    return project_root / candidate_path


def load_stage_config(
    section_name: str, config_path: Path = CONFIG_PATH
) -> tuple[ConfigContext, dict[str, Any]]:
    """Load shared [paths] settings and one stage-specific config section."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as file_handle:
        config = tomllib.load(file_handle)

    paths_config = config.get("paths")
    if not isinstance(paths_config, dict):
        raise KeyError(f"Missing [paths] table in {config_path}")

    stage_config = paths_config.get(section_name)
    if not isinstance(stage_config, dict):
        raise KeyError(f"Missing [paths.{section_name}] table in {config_path}")

    project_root = Path(get_required_string(paths_config, "project_root", config_path))
    if not project_root.is_absolute():
        raise ValueError(f"Expected 'project_root' in {config_path} to be absolute.")

    project_name = get_required_string(paths_config, "project_name", config_path)
    context = ConfigContext(
        config_path=config_path,
        project_root=project_root,
        project_name=project_name,
    )
    return context, stage_config


def resolve_config_path(
    context: ConfigContext,
    relative_path: str,
    key: str,
    *,
    include_project_name: bool = False,
) -> Path:
    """Resolve one config path and optionally scope it by the active project name."""
    resolved_path = resolve_project_path(
        context.project_root,
        relative_path,
        key,
        context.config_path,
    )
    if include_project_name:
        return resolved_path / context.project_name
    return resolved_path
