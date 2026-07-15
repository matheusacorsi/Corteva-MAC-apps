from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epg_tool.models.registry import has_trained_model, load_model, model_path_for, save_model
from epg_tool.models.tabular import TabularModel, random_forest_model, xgboost_model
from epg_tool.species.profile import SpeciesProfile, WaveformDef


def _separable_dataset(n_per_class: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_a = rng.normal(loc=0.0, scale=0.1, size=(n_per_class, 2))
    X_b = rng.normal(loc=5.0, scale=0.1, size=(n_per_class, 2))
    X = np.vstack([X_a, X_b])
    y = np.array([1] * n_per_class + [4] * n_per_class)  # waveform codes, not 0/1
    return pd.DataFrame(X, columns=["f1", "f2"]), y


@pytest.mark.parametrize("factory", [random_forest_model, xgboost_model])
def test_tabular_model_fits_and_predicts_correct_codes(factory):
    X, y = _separable_dataset()
    model = factory()
    model.fit(X, y)
    preds = model.predict(X)
    assert (preds == y).mean() > 0.95
    assert set(model.classes_) == {1, 4}


def test_tabular_model_predict_proba_shape():
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_tabular_model_save_and_load_roundtrip(tmp_path):
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = TabularModel.load(path)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_tabular_model_load_from_bytes_roundtrip(tmp_path):
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = TabularModel.load_from_bytes(path.read_bytes())
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_feature_importances_indexed_by_name():
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)
    importances = model.feature_importances()
    assert set(importances.index) == {"f1", "f2"}


def _dummy_profile(tmp_registry_path) -> SpeciesProfile:
    return SpeciesProfile(
        name="dummy",
        common_name="dummy",
        reference="",
        waveforms=[WaveformDef(code=1, label="Np")],
        sentinel_codes=frozenset({99}),
        model_registry={"random_forest": str(tmp_registry_path)},
    )


def test_registry_save_and_load(tmp_path):
    registry_path = tmp_path / "models" / "rf.joblib"
    profile = _dummy_profile(registry_path)
    X, y = _separable_dataset()
    model = random_forest_model(n_estimators=10).fit(X, y)

    assert not has_trained_model(profile, "random_forest")
    saved_path = save_model(model, profile, "random_forest")
    assert saved_path == registry_path
    assert has_trained_model(profile, "random_forest")

    loaded = load_model(profile, "random_forest")
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_registry_unregistered_model_type_raises(tmp_path):
    profile = _dummy_profile(tmp_path / "rf.joblib")
    with pytest.raises(KeyError, match="xgboost"):
        model_path_for(profile, "xgboost")


def test_registry_missing_file_raises(tmp_path):
    profile = _dummy_profile(tmp_path / "does_not_exist.joblib")
    with pytest.raises(FileNotFoundError):
        load_model(profile, "random_forest")


def test_model_path_for_anchors_relative_paths_to_project_root(monkeypatch, tmp_path):
    """model_registry paths in species YAML are relative to wherever the
    epg_tool package lives on disk, not the process's CWD -- regression
    test for a bug where a bare Path(raw) resolved against CWD and
    silently missed the model when deployed in a monorepo subfolder."""
    from epg_tool.models.registry import _PROJECT_ROOT

    profile = SpeciesProfile(
        name="dummy_relative",
        common_name="dummy_relative",
        reference="",
        waveforms=[WaveformDef(code=1, label="Np")],
        sentinel_codes=frozenset({99}),
        model_registry={"random_forest": "models/diaphorina_citri/random_forest.joblib"},
    )
    monkeypatch.chdir(tmp_path)  # simulate running from an unrelated CWD
    resolved = model_path_for(profile, "random_forest")
    assert resolved == _PROJECT_ROOT / "models/diaphorina_citri/random_forest.joblib"
    assert resolved != tmp_path / "models/diaphorina_citri/random_forest.joblib"
