"""Resolves and persists trained model artifacts through a species
profile's ``model_registry`` paths, so switching species is just
switching which profile is loaded -- no code branches on species."""

from __future__ import annotations

from pathlib import Path

from epg_tool.species.profile import SpeciesProfile

from .tabular import TabularModel


def model_path_for(profile: SpeciesProfile, model_type: str) -> Path:
    raw = profile.model_registry.get(model_type)
    if not raw:
        raise KeyError(
            f"Species profile {profile.name!r} has no path registered for "
            f"model type {model_type!r}. Add one under model_registry in "
            f"its YAML file."
        )
    return Path(raw)


def load_model(profile: SpeciesProfile, model_type: str) -> TabularModel:
    path = model_path_for(profile, model_type)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained {model_type!r} model found at {path} for species "
            f"{profile.name!r} -- train one first."
        )
    return TabularModel.load(path)


def save_model(model: TabularModel, profile: SpeciesProfile, model_type: str) -> Path:
    path = model_path_for(profile, model_type)
    model.save(path)
    return path


def has_trained_model(profile: SpeciesProfile, model_type: str) -> bool:
    try:
        return model_path_for(profile, model_type).exists()
    except KeyError:
        return False
