"""Shared helpers for resolving project-scoped pipeline paths from TOML config."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_ENV = "EMERGENCE_CONFIG"


@dataclass(frozen=True)
class ConfigContext:
    """Shared context loaded from the top-level [paths] TOML table."""

    config_path: Path
    project_root: Path
    project_name: str
    time_steps: int
    data_range: int


def _normalize_config_path(candidate_path: str | Path) -> Path:
    """Return an absolute config path resolved from the current working directory."""
    resolved_path = Path(candidate_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path
    return resolved_path.resolve(strict=False)


def resolve_default_config_path(config_path: Path | None = None) -> Path:
    """Resolve the config path from an explicit value, env var, or cwd default."""
    if config_path is not None:
        return _normalize_config_path(config_path)

    env_config_path = os.environ.get(DEFAULT_CONFIG_ENV)
    if env_config_path:
        return _normalize_config_path(env_config_path)

    return _normalize_config_path(Path("config") / "paths.toml")


CONFIG_PATH: Path = resolve_default_config_path()


def get_required_string(config: dict[str, Any], key: str, config_path: Path) -> str:
    """Return a required non-empty string value from a TOML mapping."""
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected '{key}' to be a non-empty string in {config_path}.")
    return value


def get_required_int(config: dict[str, Any], key: str, config_path: Path) -> int:
    """Return a required integer value from a TOML mapping."""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected '{key}' to be an integer in {config_path}.")  # noqa: TRY004
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
    section_name: str, config_path: Path | None = None
) -> tuple[ConfigContext, dict[str, Any]]:
    """Load shared [paths] settings and one stage-specific config section."""
    resolved_config_path = resolve_default_config_path(config_path)
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    with resolved_config_path.open("rb") as file_handle:
        config = tomllib.load(file_handle)

    paths_config = config.get("paths")
    if not isinstance(paths_config, dict):
        raise KeyError(f"Missing [paths] table in {resolved_config_path}")

    stage_config = paths_config.get(section_name)
    if not isinstance(stage_config, dict):
        raise KeyError(
            f"Missing [paths.{section_name}] table in {resolved_config_path}"
        )

    project_root = Path(
        get_required_string(paths_config, "project_root", resolved_config_path)
    )
    if not project_root.is_absolute():
        raise ValueError(
            f"Expected 'project_root' in {resolved_config_path} to be absolute."
        )

    project_name = get_required_string(
        paths_config, "project_name", resolved_config_path
    )
    window_config = paths_config if "time_steps" in paths_config else stage_config
    range_config = paths_config if "data_range" in paths_config else stage_config
    time_steps = get_required_int(window_config, "time_steps", resolved_config_path)
    data_range = get_required_int(range_config, "data_range", resolved_config_path)
    context = ConfigContext(
        config_path=resolved_config_path,
        project_root=project_root,
        project_name=project_name,
        time_steps=time_steps,
        data_range=data_range,
    )

    stage_settings = dict(stage_config)
    stage_settings.setdefault("time_steps", time_steps)
    stage_settings.setdefault("data_range", data_range)
    return context, stage_settings


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


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_CONFIG_ENV",
    "ConfigContext",
    "get_required_int",
    "get_required_string",
    "load_stage_config",
    "resolve_config_path",
    "resolve_default_config_path",
    "resolve_project_path",
]
