from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import REAL_ANA, REAL_D01, make_ana_bytes, make_d0x_bytes, requires_real_data
from epg_tool.species.profile import load_profile
from epg_tool.training.dataset import (
    RecordingRef,
    build_dataset,
    discover_recordings,
    group_train_val_test_split,
)


@pytest.fixture()
def two_recordings(tmp_path):
    """Two tiny synthetic recordings under a Stylet+-like layout:
    arquivosDataAcquisition/{insect}.D01 + DataANA/{insect}_.ANA
    """
    acq_dir = tmp_path / "arquivosDataAcquisition"
    ana_dir = acq_dir / "DataANA"
    acq_dir.mkdir()
    ana_dir.mkdir()

    for insect in ["insectA", "insectB"]:
        samples = list(np.sin(np.linspace(0, 20, 200)) * 0.1)
        (acq_dir / f"{insect}.D01").write_bytes(make_d0x_bytes(samples, sample_rate_hz="100,000"))
        rows = [(1, "0,00", 0), (2, "1,00", 0), (99, "2,00", 0)]
        (ana_dir / f"{insect}_.ANA").write_bytes(make_ana_bytes(rows))

    return tmp_path


def test_discover_recordings_finds_both(two_recordings):
    recordings = discover_recordings(two_recordings)
    assert {r.insect_id for r in recordings} == {"insectA", "insectB"}
    for rec in recordings:
        assert rec.d0x_paths[0].name == f"{rec.insect_id}.D01"
        assert rec.ana_path.name == f"{rec.insect_id}_.ANA"


def test_discover_recordings_raises_if_no_match(tmp_path):
    ana_dir = tmp_path / "DataANA"
    ana_dir.mkdir()
    (ana_dir / "orphan_.ANA").write_bytes(make_ana_bytes([(1, "0,00", 0), (99, "1,00", 0)]))
    with pytest.raises(FileNotFoundError):
        discover_recordings(tmp_path)


def test_build_dataset_groups_by_insect_id(two_recordings):
    profile = load_profile("diaphorina_citri")
    recordings = discover_recordings(two_recordings)
    X, y, groups = build_dataset(recordings, profile, window_s=0.5)
    assert len(X) == len(y) == len(groups)
    assert set(groups) == {"insectA", "insectB"}
    assert isinstance(X, pd.DataFrame)


def test_group_split_keeps_insects_disjoint():
    rng = np.random.default_rng(0)
    n_per_insect = 20
    insects = [f"insect{i}" for i in range(5)]
    X = pd.DataFrame({"f1": rng.normal(size=n_per_insect * len(insects))})
    y = rng.integers(1, 3, size=n_per_insect * len(insects))
    groups = np.repeat(insects, n_per_insect)

    split = group_train_val_test_split(X, y, groups, test_size=0.2, val_size=0.2, random_state=0)
    train_groups = set(split.train[2])
    val_groups = set(split.val[2])
    test_groups = set(split.test[2])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert train_groups | val_groups | test_groups == set(insects)


def test_group_split_falls_back_and_warns_with_few_insects():
    X = pd.DataFrame({"f1": np.arange(100.0)})
    y = np.zeros(100, dtype=int)
    groups = np.array(["only_insect"] * 100)

    with pytest.warns(UserWarning, match="Only 1 insect"):
        split = group_train_val_test_split(X, y, groups, test_size=0.2, val_size=0.2)

    assert len(split.train[0]) + len(split.val[0]) + len(split.test[0]) == 100
    # chronological: train comes first, test comes last
    assert split.train[0]["f1"].max() < split.val[0]["f1"].min()
    assert split.val[0]["f1"].max() < split.test[0]["f1"].min()


@requires_real_data
def test_discover_recordings_against_real_data_dir():
    data_root = REAL_ANA.parent.parent.parent  # "Arquivo Ondas EPG Diaphorina citri"
    recordings = discover_recordings(data_root)
    assert any(r.insect_id == "PsilideoMudaT1-3-ch6" for r in recordings)
    rec = next(r for r in recordings if r.insect_id == "PsilideoMudaT1-3-ch6")
    assert rec.ana_path == REAL_ANA
    assert rec.d0x_paths[0] == REAL_D01
    assert len(rec.d0x_paths) == 9
