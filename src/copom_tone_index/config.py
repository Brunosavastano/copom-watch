from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


@dataclass(frozen=True)
class ProjectPaths:
    database: Path
    raw: Path
    processed: Path
    figures: Path
    reports: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def load_settings() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "settings.yaml")


def load_v2_settings() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "v2_settings.yaml")


def load_topics() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "topic_taxonomy.yaml")["topics"]


def load_lexicon() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "hawkish_dovish_lexicon.yaml")


def get_paths(settings: dict[str, Any] | None = None) -> ProjectPaths:
    settings = settings or load_settings()
    path_settings = settings["paths"]
    return ProjectPaths(
        database=ROOT / path_settings["database"],
        raw=ROOT / path_settings["raw"],
        processed=ROOT / path_settings["processed"],
        figures=ROOT / path_settings["figures"],
        reports=ROOT / path_settings["reports"],
    )


def ensure_directories(paths: ProjectPaths) -> None:
    for path in [paths.database.parent, paths.raw, paths.processed, paths.figures, paths.reports]:
        path.mkdir(parents=True, exist_ok=True)
