"""Wearable windows with label roles, the frozen-trunk fine-tune, and scripts/finetune.py."""

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from uaar.data.wearable_windows import Roles, session_windows, subject_windows, validation_split
from uaar.models.edl import EDLMethod
from uaar.models.finetune import finetune, frozen_trunk
from uaar.models.softmax import SoftmaxMethod

CLASSES = ["Jumping Jacks", "Lateral Raise", "Shoulder Press (dumbbell)", "Curl long name"]
ROLES = Roles(
    known={
        "curl": "Curl long name",
        "shoulder_press": "Shoulder Press (dumbbell)",
        "lateral_raise": "Lateral Raise",
        "jumping_jacks": "Jumping Jacks",
    },
    background=["standing_still", "arm_circles"],
    unknown=["waving"],
    shifted={"curl_loose": "Curl long name"},
)
# label -> (axis, frequency in Hz): every movement is a sinusoid on its own axis and rate
PATTERN = {
    "curl": (0, 0.5),
    "shoulder_press": (1, 0.8),
    "lateral_raise": (2, 1.1),
    "jumping_jacks": (0, 2.5),
    "standing_still": (1, 0.0),
    "arm_circles": (2, 3.0),
    "waving": (1, 4.0),
    "curl_loose": (0, 0.5),
}


def write_session(root, subject, session, sets, seconds=12, seed=0):
    """A session folder in the recorder's format; ``sets`` is a list of labels in order."""
    rng = np.random.default_rng(seed)
    d = root / subject / session
    d.mkdir(parents=True)
    rows, events, t_ms, seq = [], [], 1000, 0
    for k, label in enumerate(sets):
        axis, hz = PATTERN[label]
        for phase, n in (("rest", 50), ("set", seconds * 50)):
            start_ms, start_seq = t_ms, seq
            for i in range(n):
                acc = rng.normal(0, 40, 3)
                acc[2] += 4096
                gyr = rng.normal(0, 20, 3)
                if phase == "set":
                    acc[axis] += 2500 * np.sin(2 * np.pi * hz * i / 50)
                    gyr[axis] += 1500 * np.cos(2 * np.pi * hz * i / 50)
                flags = 2 if phase == "set" else 0
                rows.append(
                    [t_ms, *np.round(acc).astype(int), *np.round(gyr).astype(int), flags, seq]
                )
                t_ms += 20
                seq = (seq + 1) % 65536
            if phase == "set":
                events.append([k, label, k + 1, start_ms, t_ms - 20, start_seq, seq - 1, n, 0])
    lines = ["# UAAR fw 0.1.0 name=UAAR-TEST", "t_ms,ax,ay,az,gx,gy,gz,flags,seq"]
    lines += [",".join(str(int(v)) for v in r) for r in rows]
    (d / "raw.csv").write_text("\n".join(lines) + "\n")
    cols = "set_index,label,session_id,t_start_ms,t_end_ms,seq_start,seq_end,n_samples,n_dropped"
    (d / "events.csv").write_text(
        cols + "\n" + "\n".join(",".join(str(v) for v in e) for e in events) + "\n"
    )
    (d / "meta.json").write_text(json.dumps({"subject": subject, "session": session}))
    return d


KNOWN_SETS = ["curl", "shoulder_press", "lateral_raise", "jumping_jacks"]
SESSION = KNOWN_SETS * 2 + ["standing_still", "arm_circles", "waving", "curl_loose"]


@pytest.fixture(scope="module")
def recordings(tmp_path_factory):
    root = tmp_path_factory.mktemp("wearable")
    write_session(root, "W01", "s1", SESSION, seed=1)
    write_session(root, "W02", "s1", SESSION, seed=2)
    return root


def test_roles_and_session_windows(recordings):
    assert ROLES.of("curl") == ("known", "Curl long name")
    assert ROLES.of("Lateral Raise") == ("known", "Lateral Raise")  # class names map to themselves
    assert ROLES.of("curl_loose") == ("shifted", "Curl long name")
    assert ROLES.of("arm_circles") == ("background", "") and ROLES.of("waving") == ("unknown", "")
    assert ROLES.of("typo") == ("", "")

    X, t = session_windows(recordings / "W01" / "s1")
    assert X.shape[1:] == (6, 100) and len(X) == len(t) and (t["label"] != "").all()
    assert (t["set_index"] >= 0).all() and t["set_index"].nunique() == len(SESSION)
    # 12 s sets, 2 s windows, 1 s hop: 11 windows inside a set, a few more touch its edges
    per_set = t.groupby("set_index").size()
    assert per_set.min() >= 11 and per_set.max() <= 13
    assert abs(float(X[:, 2].mean()) - 1.0) < 0.05  # gravity on z, in g


def test_subject_windows_and_validation_split(recordings, tmp_path):
    X, t = subject_windows([recordings / "W01"], ROLES)
    assert set(t["role"]) == {"known", "background", "unknown", "shifted"} and len(X) == len(t)
    assert t.attrs["ignored"] == {}
    train, val = validation_split(t)
    assert not (train & val).any()
    curl = (t["label"] == "curl").to_numpy()
    # two sets: the second one is validation, the first is training
    assert set(t.loc[curl & val, "set_index"]) == {4} and set(t.loc[curl & train, "set_index"]) == {
        0
    }
    # one set: its tail is validation and the window overlapping both parts is in neither
    wave = (t["label"] == "waving").to_numpy()
    assert 0 < (wave & val).sum() < (wave & train).sum()
    assert (wave & ~train & ~val).sum() == 1

    other = Roles(known={"curl": "Curl long name"})
    _, t2 = subject_windows([recordings / "W01"], other)
    assert set(t2["label"]) == {"curl"} and t2.attrs["ignored"]["waving"] > 0
    with pytest.raises(FileNotFoundError):
        subject_windows([tmp_path], ROLES)


def _encode(table):
    index = {c: i for i, c in enumerate(CLASSES)}
    return np.array([index.get(c, -1) for c in table["class_name"]])


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """Taught softmax and EDL models trained on a third "person" (the RecoFit stand-in), whose
    movements are a little slower than those of W01 and W02."""
    from uaar.config import Config

    root = tmp_path_factory.mktemp("source")
    slower = {k: (axis, hz * 0.85) for k, (axis, hz) in PATTERN.items()}
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(globals(), "PATTERN", slower)
        write_session(root, "SRC", "s1", SESSION, seconds=40, seed=9)
    X, t = subject_windows([root / "SRC"], ROLES)
    train, val = validation_split(t)
    y = _encode(t)
    known, bg = (t["role"] == "known").to_numpy(), (t["role"] == "background").to_numpy()
    methods = {}
    for name, cls in (("softmax", SoftmaxMethod), ("edl", EDLMethod)):
        over = ["train.epochs=12", "train.batch_size=32", "train.patience=12"]
        cfg = Config.load(f"configs/k4taught/{name}.yaml", over)
        torch.manual_seed(0)
        m = cls(len(CLASSES), CLASSES, background="uncertain")
        m.fit(
            X[known & train], y[known & train], X[known & val], y[known & val], cfg, root,
            background=(X[bg & train], X[bg & val]),
        )  # fmt: skip
        methods[name] = m
    run = root / "k4taught_x_softmax_fold0_bg0_seed0"
    methods["softmax"].save(run)
    (run / "threshold.json").write_text(json.dumps({"tau": {"msp": 0.5, "entropy": 1.0}}))
    return {"run": run, "edl": methods["edl"]}


def test_frozen_trunk_freezes_weights_and_batchnorm():
    net = SoftmaxMethod(4, CLASSES).model
    with frozen_trunk(net):
        assert not any(p.requires_grad for p in net.features.parameters())
        assert all(p.requires_grad for p in net.head.parameters())
        net.train()
        assert net.training and not net.features.training
    assert all(p.requires_grad for p in net.features.parameters())
    net.train()
    assert net.features.training  # the override is gone


def test_finetune_edl_changes_only_the_output_layers(recordings, source):
    X, t = subject_windows([recordings / "W01"], ROLES)
    train, val = validation_split(t)
    y = _encode(t)
    known, bg = (t["role"] == "known").to_numpy(), (t["role"] == "background").to_numpy()
    m = source["edl"]
    old_stats = dict(m.stats)
    trunk = {k: v.clone() for k, v in m.model.features.state_dict().items()}
    head = m.model.head.weight.detach().clone()
    h = finetune(
        m, X[known & train], y[known & train], X[known & val], y[known & val], X[bg & train],
        epochs=15, patience=15, batch_size=16,
    )  # fmt: skip
    for k, v in m.model.features.state_dict().items():
        torch.testing.assert_close(v, trunk[k])  # weights and BatchNorm running statistics
    assert not torch.equal(m.model.head.weight, head)
    assert m.stats != old_stats  # normalization re-fitted on the wearable windows
    assert h.best_val_f1 > 0.8
    assert (m.predict(X[known & val]).label == y[known & val]).mean() > 0.8

    other = SoftmaxMethod(len(CLASSES), CLASSES, background="other")
    with pytest.raises(ValueError, match="Other"):
        finetune(other, X[known & train], y[known & train], X[known & val], y[known & val])


def test_finetune_script_end_to_end(recordings, source, tmp_path, monkeypatch, capsys):
    from uaar.live.predictors import RunPredictor

    roles = tmp_path / "wearable.yaml"
    roles.write_text(
        json.dumps(  # JSON is YAML
            {
                "known": ROLES.known,
                "background": ROLES.background,
                "unknown": ROLES.unknown,
                "shifted": ROLES.shifted,
            }
        )
    )
    run = source["run"]
    spec = importlib.util.spec_from_file_location("finetune_script", "scripts/finetune.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    argv = [
        "finetune.py", "--run", str(run), "--train", str(recordings / "W01"),
        "--test", str(recordings / "W02"), "--roles", str(roles), "--epochs", "15",
        "--patience", "15", "--batch-size", "16", "--T", "5", "--out", str(tmp_path / "ft"),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    assert script.main() == 0

    out, mc = tmp_path / "ft", tmp_path / "ft-mcdropout"
    for d in (out, mc):
        assert {
            "model.pt",
            "config.json",
            "preprocess.json",
            "threshold.json",
            "finetune.json",
        } <= {p.name for p in d.iterdir()}
    info = json.loads((out / "finetune.json").read_text())
    assert info["train_subjects"] == ["W01"] and info["counts"]["train_background"] > 0
    assert set(json.loads((out / "threshold.json").read_text())["tau"]) == {"msp", "entropy"}
    rows = pd.DataFrame(json.loads((out / "report.json").read_text()))
    assert rows["model"].tolist() == [
        "softmax, RecoFit only", "softmax, fine-tuned", "mcdropout, fine-tuned",
    ]  # fmt: skip
    assert rows["score"].tolist() == ["msp", "msp", "entropy"]
    tuned = rows.iloc[1]
    assert tuned["accuracy"] > 0.9
    for col in ("known_kept", "unknown_accepted", "background_accepted", "shifted_accuracy"):
        assert 0.0 <= tuned[col] <= 1.0
    assert "| softmax, fine-tuned | msp |" in (out / "report.md").read_text()
    assert "Trained on W01, tested on W02" in capsys.readouterr().out

    # the live page can load both fine-tuned runs
    X = np.zeros((6, 100), dtype=np.float32)
    assert RunPredictor(out).predict(X)["label"] in CLASSES
    assert RunPredictor(mc, score="entropy").predict(X)["label"] in CLASSES
