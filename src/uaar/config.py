"""Experiment configuration: base.yaml <- method yaml <- dotted command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from uaar.utils import repo_root


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _parse_value(text: str) -> Any:
    """``--set`` values are parsed as YAML scalars so ``true``, ``1e-3`` and ``[a,b]`` work."""
    return yaml.safe_load(text)


class Config:
    """A nested dict with attribute access (``cfg.train.lr``) and provenance.

    ``Config.load(method_file, overrides)`` merges ``configs/base.yaml`` with the method file and
    ``key.path=value`` overrides, in that order. ``to_yaml()`` writes the fully resolved result.
    """

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            v = self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(v) if isinstance(v, dict) else v

    def __getitem__(self, key: str) -> Any:
        v = self._data[key]
        return Config(v) if isinstance(v, dict) else v

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        cur = self._data
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def to_yaml(self, path: Path | None = None) -> str:
        text = yaml.safe_dump(self._data, sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    @classmethod
    def load(
        cls,
        method_file: Path | str | None = None,
        overrides: list[str] | None = None,
        base: Path | str | None = None,
    ) -> Config:
        base = Path(base) if base else repo_root() / "configs" / "base.yaml"
        data = yaml.safe_load(Path(base).read_text()) or {}
        if method_file is not None:
            _deep_update(data, yaml.safe_load(Path(method_file).read_text()) or {})
        cfg = cls(data)
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"override must be key.path=value, got {item!r}")
            k, v = item.split("=", 1)
            cfg.set(k.strip(), _parse_value(v.strip()))
        return cfg

    @classmethod
    def from_yaml(cls, path: Path | str) -> Config:
        return cls(yaml.safe_load(Path(path).read_text()) or {})
