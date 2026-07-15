"""Species profile: waveform label set, rule-based thresholds, and model
registry for one insect/species, loaded from a YAML config file.

This is the extensibility point requested for the tool -- adding a new
insect (aphid, leafhopper, ...) means writing a new YAML file under
``species/profiles/``, not touching any parsing/feature/model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_PROFILES_DIR = Path(__file__).parent / "profiles"


@dataclass(frozen=True)
class WaveformDef:
    code: int
    label: str
    description: str = ""
    color: str = "#888888"


@dataclass
class SpeciesProfile:
    name: str
    common_name: str
    reference: str
    waveforms: list[WaveformDef]
    sentinel_codes: frozenset[int]
    rule_based_thresholds: dict = field(default_factory=dict)
    model_registry: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        codes = [w.code for w in self.waveforms]
        if len(codes) != len(set(codes)):
            raise ValueError(f"{self.name}: duplicate waveform codes in {codes}")
        overlap = set(codes) & set(self.sentinel_codes)
        if overlap:
            raise ValueError(f"{self.name}: codes {overlap} are both waveforms and sentinels")

    @property
    def code_to_label(self) -> dict[int, str]:
        return {w.code: w.label for w in self.waveforms}

    @property
    def label_to_code(self) -> dict[str, int]:
        return {w.label: w.code for w in self.waveforms}

    def label_for_code(self, code: int) -> str | None:
        return self.code_to_label.get(code)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpeciesProfile":
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        waveforms = [
            WaveformDef(
                code=w["code"],
                label=w["label"],
                description=w.get("description", ""),
                color=w.get("color", "#888888"),
            )
            for w in raw.get("waveforms", [])
        ]
        return cls(
            name=raw["name"],
            common_name=raw.get("common_name", raw["name"]),
            reference=raw.get("reference", ""),
            waveforms=waveforms,
            sentinel_codes=frozenset(raw.get("sentinel_codes", [99])),
            rule_based_thresholds=raw.get("rule_based_thresholds", {}),
            model_registry=raw.get("model_registry", {}),
            parameters=raw.get("parameters", {}),
        )


def list_profiles() -> list[str]:
    """Names of every built-in species profile (YAML filename stems)."""
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))


def load_profile(name: str) -> SpeciesProfile:
    """Load a built-in profile by name (matches the YAML filename stem)."""
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No species profile named {name!r}. Available: {available}")
    return SpeciesProfile.from_yaml(path)
