from pathlib import Path

from uaar.config import Config


def test_merge_and_overrides(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("method: softmax\ntrain: {epochs: 60, lr: 1.0e-3}\nmodel: {dropout: 0.3}\n")
    method = tmp_path / "edl.yaml"
    method.write_text("method: edl\nmodel: {edl: {evidence: softplus}}\n")
    cfg = Config.load(
        method, ["train.epochs=2", "model.dropout=0.5", "train.extra=true"], base=base
    )
    assert cfg.method == "edl" and cfg.train.epochs == 2 and cfg.train.lr == 1e-3
    assert (
        cfg.model.dropout == 0.5
        and cfg.model.edl.evidence == "softplus"
        and cfg.get("train.extra") is True
    )
    out = tmp_path / "resolved.yaml"
    cfg.to_yaml(out)
    assert Config.from_yaml(out).to_dict() == cfg.to_dict()


def test_repo_configs_resolve():
    for name in ("softmax", "edl", "logreg", "mcdropout"):
        cfg = Config.load(Path("configs/p1") / f"{name}.yaml")
        assert cfg.method == name and cfg.data.window_s == 2.0
