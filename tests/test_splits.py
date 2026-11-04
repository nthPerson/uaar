"""Split invariants (spec §7, test_splits row) on a synthetic window meta frame."""

import json

import numpy as np
import pandas as pd
import pytest

from uaar.data import splits as sp

N_SUBJECTS = 25
# label -> (number of subjects that have it, windows per subject)
CLASSES = {
    "Curl": (25, 40),
    "Squat": (25, 40),
    "Pushup": (25, 40),
    "Row": (22, 40),
    "Lunge": (20, 40),
    "Jump": (20, 15),  # exactly 20 subjects, 300 windows = 5.0 min at 1 s hop
    "Rare": (5, 100),  # too few subjects
    "Short": (25, 5),  # 125 windows = 2.1 min: too short
}
NONEX = ["Rest", "Transition"]


def _meta() -> pd.DataFrame:
    rows = []
    subjects = [f"{i:02d}" for i in range(N_SUBJECTS)]
    for label, (n_sub, per_sub) in CLASSES.items():
        for s in subjects[:n_sub]:
            for j in range(per_sub):
                rows.append((s, f"single/{s}/{label}", "single", label, float(j)))
    # Non-exercise windows live in "multi" recordings; a few "single" rows with the same label
    # must never be selected by select_nonexercise.
    for s in subjects:
        for label in NONEX:
            for j in range(3):
                rows.append((s, f"multi/{s}", "multi", label, float(j)))
            rows.append((s, f"single/{s}/x", "single", label, 0.0))
    meta = pd.DataFrame(rows, columns=["subject", "recording_id", "source", "label", "start_s"])
    meta.insert(0, "window_id", np.arange(len(meta), dtype=np.int64))
    meta["label_purity"] = np.float32(1.0)
    meta["n_gaps"] = 0
    return meta


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    return _meta()


@pytest.fixture(scope="module")
def eligible(meta) -> list[str]:
    return sp.eligible_classes(meta, min_subjects=20, min_minutes=5.0)


# ---- Eligibility ---------------------------------------------------------------------------
def test_eligibility_thresholds(meta, eligible):
    # Rest/Transition have 25 subjects but only 100 windows (1.7 min): not eligible.
    assert eligible == sorted(["Curl", "Squat", "Pushup", "Row", "Lunge", "Jump"])
    # Jump sits exactly on both thresholds (20 subjects, 5.0 min): included.
    assert "Jump" in eligible
    assert "Jump" not in sp.eligible_classes(meta, min_subjects=21, min_minutes=5.0)
    assert "Jump" not in sp.eligible_classes(meta, min_subjects=20, min_minutes=5.01)
    assert "Rare" not in eligible and "Short" not in eligible
    # Minutes are n_windows * hop_s / 60: a 2 s hop doubles the minutes.
    assert "Short" in sp.eligible_classes(meta, min_subjects=20, min_minutes=4.0, hop_s=2.0)
    assert "" not in sp.eligible_classes(
        pd.DataFrame({"subject": ["a"] * 10, "label": [""] * 10}), min_subjects=1, min_minutes=0
    )


# ---- Class splits --------------------------------------------------------------------------
def test_make_class_split_is_seeded_sorted_and_disjoint(eligible):
    known, unknown = sp.make_class_split(eligible, frac_unknown=0.25, seed=0)
    assert known == sorted(known) and unknown == sorted(unknown)
    assert set(known) | set(unknown) == set(eligible)
    assert not set(known) & set(unknown)
    assert len(unknown) == round(0.25 * len(eligible)) >= 1
    assert sp.make_class_split(eligible, 0.25, seed=0) == (known, unknown)
    other = sp.make_class_split(eligible, 0.25, seed=1)
    assert len(other[1]) == len(unknown)
    # A tiny fraction still holds out at least one class.
    assert len(sp.make_class_split(eligible, frac_unknown=0.0, seed=0)[1]) == 1


def test_near_split_uses_only_eligible_classes(eligible):
    known, unknown = sp.near_unknown_split(eligible, {"Row": "Curl", "Lunge": "Squat"})
    assert unknown == ["Lunge", "Row"]
    assert known == sorted(set(eligible) - {"Lunge", "Row"})
    with pytest.raises(ValueError):
        sp.near_unknown_split(eligible, {"Rare": "Curl"})  # unknown not eligible
    with pytest.raises(ValueError):
        sp.near_unknown_split(eligible, {"Row": "Short"})  # known not eligible
    with pytest.raises(ValueError):
        sp.near_unknown_split(eligible, {"Row": "Curl", "Curl": "Squat"})  # both roles


# ---- Subject folds -------------------------------------------------------------------------
def test_subject_folds_partition_subjects(meta):
    folds = sp.make_subject_folds(meta, n_folds=5, val_frac=0.15, seed=0)
    assert len(folds) == 5
    all_subjects = set(meta["subject"])
    tested = []
    for f in folds:
        tr, va, te = map(set, (f["train_subjects"], f["val_subjects"], f["test_subjects"]))
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert tr | va | te == all_subjects
        assert len(te) == 5
        assert len(va) == round(0.15 * 20) == 3
        tested.extend(te)
    assert sorted(tested) == sorted(all_subjects)  # every subject tested exactly once
    assert sp.make_subject_folds(meta, 5, 0.15, seed=0) == folds
    assert sp.make_subject_folds(meta, 5, 0.15, seed=1) != folds


# ---- Written splits and leakage ------------------------------------------------------------
@pytest.fixture(scope="module")
def written(meta, tmp_path_factory):
    # <...>/recofit/splits: the dataset name defaults to the parent directory name.
    out = tmp_path_factory.mktemp("data") / "recofit" / "splits"
    paths = sp.write_splits(
        meta,
        out,
        n_folds=5,
        class_split_seeds=(0, 1, 2),
        near_pairs={"Row": "Curl", "Lunge": "Squat"},
        min_subjects=20,
        min_minutes=5.0,
    )
    return paths


def test_write_splits_files_and_schema(written):
    names = sorted(p.name for p in written)
    assert len(names) == 5 * 4
    assert names[0] == "fold0_near.json" and "fold4_rand2.json" in names
    d = json.loads(written[0].read_text())
    assert list(d) == [
        "dataset",
        "fold",
        "n_folds",
        "class_split",
        "class_split_seed",
        "train_subjects",
        "val_subjects",
        "test_subjects",
        "known_classes",
        "unknown_classes",
        "eligibility",
        "created_by",
    ]
    assert d["dataset"] == "recofit"
    assert d["eligibility"] == {"min_subjects": 20, "min_minutes": 5.0}
    assert isinstance(d["created_by"], str) and d["created_by"]


def test_write_splits_round_trips_through_load_split(written):
    for path in written:
        split = sp.load_split(path)
        assert isinstance(split, sp.Split)
        assert split.to_dict() == json.loads(path.read_text())
        assert path.stem == split.split_id
        assert split.known_classes == sorted(split.known_classes)
        if split.class_split == "near":
            assert split.class_split_seed is None
            assert split.unknown_classes == ["Lunge", "Row"]
        else:
            assert split.class_split_seed == int(split.class_split[-1])


def test_no_subject_in_two_parts_and_no_unknown_in_train_or_val(meta, written):
    for path in written:
        split = sp.load_split(path)
        parts = {p: sp.select(meta, split, p) for p in sp.PARTS}
        subj = {p: set(meta.set_index("window_id").loc[ids, "subject"]) for p, ids in parts.items()}
        assert not (subj["train"] & subj["val"])
        assert not (subj["train"] & subj["test_known"])
        assert not (subj["val"] & subj["test_known"])
        assert subj["test_known"] <= set(split.test_subjects)
        assert subj["test_unknown"] <= set(split.test_subjects)
        labels = meta.set_index("window_id")["label"]
        for p in ("train", "val", "test_known"):
            assert set(labels.loc[parts[p]]) <= set(split.known_classes)
            assert not set(labels.loc[parts[p]]) & set(split.unknown_classes)
        assert set(labels.loc[parts["test_unknown"]]) <= set(split.unknown_classes)
        assert len(parts["test_unknown"]) > 0 and len(parts["train"]) > 0
        # Window ids are unique across the four parts.
        ids = np.concatenate(list(parts.values()))
        assert len(ids) == len(set(ids))


def test_test_windows_only_from_test_subjects(meta, written):
    split = sp.load_split(written[0])
    test_ids = np.concatenate(
        [sp.select(meta, split, "test_known"), sp.select(meta, split, "test_unknown")]
    )
    rows = meta.set_index("window_id").loc[test_ids]
    assert set(rows["subject"]) == set(split.test_subjects)
    assert set(rows["label"]) <= set(split.known_classes) | set(split.unknown_classes)


def test_select_rejects_unknown_part(meta, written):
    with pytest.raises(ValueError):
        sp.select(meta, sp.load_split(written[0]), "test")  # type: ignore[arg-type]


def test_encoding_stable_across_folds(written):
    by_class_split: dict[str, list[sp.Split]] = {}
    for path in written:
        s = sp.load_split(path)
        by_class_split.setdefault(s.class_split, []).append(s)
    probe = np.array(["Curl", "Squat", "Row", "Rare", "", "Nope"], dtype=object)
    for name, group in by_class_split.items():
        assert len(group) == 5, name
        ref = sp.encode_labels(probe, group[0])
        for s in group[1:]:
            assert s.known_classes == group[0].known_classes
            np.testing.assert_array_equal(sp.encode_labels(probe, s), ref)
        assert ref.dtype == np.int64
        assert ref[3] == -1 and ref[4] == -1 and ref[5] == -1
        for i, c in enumerate(probe[:3]):
            if c in group[0].known_classes:
                assert ref[i] == group[0].known_classes.index(c)
            else:
                assert ref[i] == -1
    # encode_labels accepts a pandas Series too.
    s0 = by_class_split["rand0"][0]
    np.testing.assert_array_equal(
        sp.encode_labels(pd.Series(probe), s0), sp.encode_labels(probe, s0)
    )


def test_select_nonexercise_only_multi_source_test_subjects(meta, written):
    split = sp.load_split(written[0])
    ids = sp.select_nonexercise(meta, split, ["Rest", "Transition"])
    rows = meta.set_index("window_id").loc[ids]
    assert len(ids) == len(split.test_subjects) * 2 * 3
    assert set(rows["subject"]) == set(split.test_subjects)
    assert (rows["source"] == "multi").all()
    assert set(rows["label"]) == {"Rest", "Transition"}
    assert len(sp.select_nonexercise(meta, split, ["Nope"])) == 0


def test_eligible_classes_exclude():
    meta = pd.DataFrame(
        {
            "subject": [f"S{i}" for i in range(40)] * 2,
            "label": ["Non-Exercise"] * 40 + ["Curl"] * 40,
        }
    )
    assert sp.eligible_classes(meta, 2, 0.1) == ["Curl", "Non-Exercise"]
    assert sp.eligible_classes(meta, 2, 0.1, exclude=["Non-Exercise"]) == ["Curl"]
