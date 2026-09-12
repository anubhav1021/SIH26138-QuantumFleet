"""Persistence for user-created scenarios and vessels, layered on top of the
built-in `configs/` files without ever modifying them. Built-in scenarios
stay under `configs/*.yaml` (read-only from here); user scenarios live in a
separate `configs/user_scenarios/` directory; user-added vessel classes live
in a separate `configs/user_vessels.yaml`, merged on top of (never
replacing) the built-in vessel catalog. This is what "protect built-ins from
accidental modification" means concretely: a different location, not a
permission flag to remember to check.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from quantumfleet.optimization.problem import ScenarioConfig, parse_scenario
from quantumfleet.prediction.physics_model import VesselClass, parse_vessel_catalog, parse_vessel_class
from quantumfleet.utils.io import load_yaml

BUILTIN_SCENARIO_STEMS = {"default_scenario", "demo_case_study"}


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "scenario"


@dataclass(frozen=True)
class ScenarioSummary:
    key: str
    name: str
    is_builtin: bool
    path: Path
    n_routes: int


class ScenarioRepository:
    def __init__(self, builtin_dir: Path | str, user_dir: Path | str):
        self.builtin_dir = Path(builtin_dir)
        self.user_dir = Path(user_dir)
        self.user_dir.mkdir(parents=True, exist_ok=True)

    def _builtin_files(self) -> list[Path]:
        return sorted(p for p in self.builtin_dir.glob("*.yaml") if p.stem in BUILTIN_SCENARIO_STEMS)

    def _user_files(self) -> list[Path]:
        return sorted(self.user_dir.glob("*.yaml"))

    def _summarize(self, path: Path, is_builtin: bool) -> ScenarioSummary:
        raw = load_yaml(path) or {}
        return ScenarioSummary(key=path.stem, name=raw.get("name", path.stem), is_builtin=is_builtin, path=path, n_routes=len(raw.get("routes") or []))

    def list_builtin(self) -> list[ScenarioSummary]:
        return [self._summarize(p, is_builtin=True) for p in self._builtin_files()]

    def list_user(self) -> list[ScenarioSummary]:
        return [self._summarize(p, is_builtin=False) for p in self._user_files()]

    def list_all(self) -> list[ScenarioSummary]:
        return self.list_builtin() + self.list_user()

    def _resolve(self, key: str) -> tuple[Path, bool]:
        user_path = self.user_dir / f"{key}.yaml"
        if user_path.exists():
            return user_path, False
        builtin_path = self.builtin_dir / f"{key}.yaml"
        if builtin_path.exists() and key in BUILTIN_SCENARIO_STEMS:
            return builtin_path, True
        raise KeyError(f"no scenario named '{key}'")

    def load_raw(self, key: str) -> dict:
        path, _ = self._resolve(key)
        return load_yaml(path)

    def load_scenario_config(self, key: str) -> ScenarioConfig:
        return parse_scenario(self.load_raw(key))

    def is_builtin(self, key: str) -> bool:
        return self._resolve(key)[1]

    def unique_key(self, desired_name: str) -> str:
        base = slugify(desired_name)
        existing = {s.key for s in self.list_all()}
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def save_user_scenario(self, key: str, raw: dict) -> Path:
        """Create or overwrite a user scenario under `key`. Refuses to shadow
        a built-in scenario's exact key -- built-ins can only be *duplicated*
        (which allocates a fresh, non-colliding key), never overwritten."""
        if key in BUILTIN_SCENARIO_STEMS and not (self.user_dir / f"{key}.yaml").exists():
            raise ValueError(f"'{key}' is a built-in scenario name -- duplicate it to a new name instead of saving over it.")
        path = self.user_dir / f"{key}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        return path

    def duplicate(self, source_key: str, new_name: str) -> str:
        raw = dict(self.load_raw(source_key))
        raw["name"] = new_name
        new_key = self.unique_key(new_name)
        self.save_user_scenario(new_key, raw)
        return new_key

    def delete_user_scenario(self, key: str) -> None:
        path = self.user_dir / f"{key}.yaml"
        if not path.exists():
            raise KeyError(f"'{key}' is not a user-created scenario (or does not exist) -- built-in scenarios cannot be deleted.")
        path.unlink()


class VesselCatalogRepository:
    def __init__(self, builtin_path: Path | str, user_path: Path | str):
        self.builtin_path = Path(builtin_path)
        self.user_path = Path(user_path)

    def load_builtin(self) -> tuple[dict[str, VesselClass], list[float]]:
        return parse_vessel_catalog(load_yaml(self.builtin_path))

    def load_user_raw(self) -> list[dict]:
        if not self.user_path.exists():
            return []
        raw = load_yaml(self.user_path) or {}
        return raw.get("vessel_classes") or []

    def load_merged(self) -> tuple[dict[str, VesselClass], list[float]]:
        """Built-in catalog with any user-added vessel classes layered on
        top. Speed bins stay builtin-only (a QEA-wide encoding parameter,
        not a per-vessel one) -- see Scenario Builder validation for the
        check that a new vessel's speed range actually overlaps one."""
        vessel_classes, speed_bins = self.load_builtin()
        vessel_classes = dict(vessel_classes)
        for entry in self.load_user_raw():
            vessel_classes[entry["id"]] = parse_vessel_class(entry)
        return vessel_classes, speed_bins

    def is_builtin(self, vessel_id: str) -> bool:
        builtin, _ = self.load_builtin()
        return vessel_id in builtin

    def save_user_vessel(self, entry: dict) -> None:
        if self.is_builtin(entry["id"]):
            raise ValueError(f"'{entry['id']}' is a built-in vessel class id -- choose a different id.")
        entries = [e for e in self.load_user_raw() if e["id"] != entry["id"]]  # upsert
        entries.append(entry)
        self._write_user(entries)

    def delete_user_vessel(self, vessel_id: str) -> None:
        entries = self.load_user_raw()
        remaining = [e for e in entries if e["id"] != vessel_id]
        if len(remaining) == len(entries):
            raise KeyError(f"'{vessel_id}' is not a user-created vessel class (or does not exist) -- built-in vessel classes cannot be deleted.")
        self._write_user(remaining)

    def _write_user(self, entries: list[dict]) -> None:
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.user_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"vessel_classes": entries}, f, sort_keys=False)
