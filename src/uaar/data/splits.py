"""Subject folds, known/unknown class splits and window selection (spec §3.3, datasets.md §6).

Leakage rules enforced here and pinned by ``tests/test_splits.py``: a subject belongs to exactly
one of train/val/test within a fold; unknown-class windows never appear in train or val; test
windows (known and unknown) come only from the fold's test subjects.

Design choices where the spec is open:

* **Minutes per class** (eligibility) are counted as ``n_windows * hop_s / 60``. With a 1 s hop
  every kept window contributes one new second of signal, so this equals the labelled minutes
  that survived the purity rule (not ``n_windows * window_s``, which would double count).
* **Fold assignment** shuffles the sorted unique subjects with ``numpy.random.default_rng(seed)``
  and ranks them; ``sklearn.model_selection.GroupKFold`` on those ranks (one group per subject,
  equal sizes) assigns rank ``r`` to fold ``r % n_folds``, so the result is deterministic and
  honours ``seed``. Validation subjects are then drawn from each fold's training subjects with
  the same generator.
* **Class-stratified folds** (``stratify_classes``, used for RecoFit since 2026-11-05). RecoFit
  is four subject cohorts, each with its own exercise list, so a class has data from only 25 to
  44 of the 94 subjects. Folds drawn without regard to that leave some classes with one or two
  validation or test subjects. With ``stratify_classes`` the subjects are assigned by iterative
  stratification (Sechidis, Tsoumakas & Vlahavas 2011) on the subject x class table: the class
  with the fewest unassigned subjects is dealt out first, each subject going to the fold that
  still lacks the most subjects of that class. The same procedure then separates validation from
  training subjects inside each fold. Only the list of exercises a subject performed is used,
  never the signals, and a subject still belongs to exactly one of train/val/test.
* The ``near`` split has no seed; its ``class_split_seed`` is stored as ``null``.
* **Class source.** RecoFit's single-exercise file is a copy of the labelled segments of its
  multi-activity file, so taking exercise windows from both sources would count every sample
  twice. ``class_source`` (``"single"`` for RecoFit) restricts eligibility counts and the
  train/val/test_known/test_unknown selections to one ``source``; ``None`` means no restriction.
  The non-exercise set always comes from ``source == "multi"``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from uaar.utils import git_sha

Part = Literal["train", "val", "test_known", "test_unknown"]
PARTS: tuple[str, ...] = ("train", "val", "test_known", "test_unknown")


@dataclass
class Split:
    """One fold × class-split combination; mirrors ``splits/<split_id>.json`` (spec §3.3)."""

    dataset: str
    fold: int
    n_folds: int
    class_split: str
    class_split_seed: int | None
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]
    known_classes: list[str]
    unknown_classes: list[str]
    eligibility: dict = field(default_factory=dict)
    class_source: str | None = None
    fold_strategy: str = "random"
    created_by: str = "unknown"

    @property
    def split_id(self) -> str:
        """``fold{k}_{classsplit}``, the file stem under ``splits/``."""
        return f"fold{self.fold}_{self.class_split}"

    @property
    def label_index(self) -> dict[str, int]:
        """Known class -> integer label (its index in the sorted ``known_classes``)."""
        return {c: i for i, c in enumerate(self.known_classes)}

    def to_dict(self) -> dict:
        """Plain dict with the JSON key order of spec §3.3."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Split:
        """Inverse of :meth:`to_dict`; unknown keys are ignored."""
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in names})

    def save(self, path: Path) -> None:
        """Write the JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


# ---- Class eligibility and class splits -----------------------------------------------------
def eligible_classes(
    meta: pd.DataFrame,
    min_subjects: int,
    min_minutes: float,
    hop_s: float = 1.0,
    exclude: Iterable[str] = (),
    source: str | None = None,
) -> list[str]:
    """Labels with enough subjects and enough windowed minutes to be a known or unknown class.

    Args:
        meta: Window metadata (spec §3.2) with ``subject`` and ``label`` columns.
        min_subjects: A class needs windows from at least this many distinct subjects.
        min_minutes: A class needs at least this many minutes, counted as
            ``n_windows * hop_s / 60`` (each window adds ``hop_s`` seconds of new signal).
        hop_s: Hop used when the windows were made, in seconds.
        exclude: Labels that are never classes, such as the non-exercise labels that form the
            separate P1-b test set (``Non-Exercise``, ``Rest``).
        source: Count only windows with this ``source`` value (``None``: all windows).

    Returns:
        Sorted list of eligible label strings (``""`` is never eligible).
    """
    excluded = {""} | set(exclude)
    labelled = meta[~meta["label"].astype(str).isin(excluded)]
    if source is not None:
        labelled = labelled[labelled["source"].astype(str) == source]
    stats = labelled.groupby("label").agg(
        n_subjects=("subject", "nunique"), n_windows=("subject", "size")
    )
    minutes = stats["n_windows"] * hop_s / 60.0
    ok = (stats["n_subjects"] >= min_subjects) & (minutes >= min_minutes)
    return sorted(str(c) for c in stats.index[ok])


def label_table(
    meta: pd.DataFrame,
    hop_s: float = 1.0,
    class_source: str | None = None,
    nonexercise: Iterable[str] = (),
) -> pd.DataFrame:
    """Per-label windows, minutes and subjects as the eligibility rule counts them (T1 input).

    Args:
        meta: Window metadata (spec §3.2).
        hop_s: Hop used when the windows were made, in seconds.
        class_source: ``source`` that class windows are taken from (``None``: all).
        nonexercise: Labels of the non-exercise set; these are counted in ``source == "multi"``.

    Returns:
        Frame with ``label``, ``minutes`` (``windows * hop_s / 60``), ``subjects``, ``windows``
        and ``set`` (``class`` or ``nonexercise``), sorted by minutes, largest first.
    """
    nonex = set(nonexercise)
    src = meta["source"].astype(str)
    lab = meta["label"].astype(str)
    is_nonex = lab.isin(nonex)
    keep = (is_nonex & (src == "multi")) | (
        ~is_nonex & (lab != "") & ((src == class_source) if class_source is not None else True)
    )
    t = (
        meta[keep]
        .groupby("label")
        .agg(subjects=("subject", "nunique"), windows=("subject", "size"))
        .reset_index()
    )
    t["minutes"] = t["windows"] * hop_s / 60.0
    t["set"] = np.where(t["label"].isin(nonex), "nonexercise", "class")
    t = t.sort_values(["minutes", "label"], ascending=[False, True]).reset_index(drop=True)
    return t[["label", "minutes", "subjects", "windows", "set"]]


def make_class_split(
    classes: Sequence[str],
    frac_unknown: float = 0.25,
    seed: int = 0,
    tied: Iterable[Sequence[str]] = (),
) -> tuple[list[str], list[str]]:
    """Randomly hold out about ``frac_unknown`` of ``classes`` as unknown.

    Args:
        classes: Eligible class names (any order; duplicates ignored).
        frac_unknown: Fraction held out; the count is ``round(frac_unknown * n)``, at least 1.
        seed: Seed for ``numpy.random.default_rng``.
        tied: Groups of classes that must land on the same side (two labels for the same
            exercise: holding one out while the other is known would make the "unknown"
            windows a known exercise). Members that are not in ``classes`` are ignored. A
            class may be in at most one group.

    Returns:
        ``(known, unknown)``, both sorted, disjoint, covering ``classes``. With ``tied`` groups
        the classes are drawn group by group in a seeded random order, skipping a group that
        would exceed the count, so the unknown count is exact unless only groups are left.
    """
    classes = sorted(set(classes))
    if len(classes) < 2:
        raise ValueError("need at least two classes to form a known/unknown split")
    n_unknown = min(len(classes) - 1, max(1, int(round(frac_unknown * len(classes)))))
    rng = np.random.default_rng(seed)
    groups = [sorted(set(g) & set(classes)) for g in tied]
    groups = [g for g in groups if len(g) > 1]
    if not groups:
        unknown = sorted(rng.choice(classes, size=n_unknown, replace=False).tolist())
        return sorted(set(classes) - set(unknown)), unknown

    in_group = [c for g in groups for c in g]
    if len(in_group) != len(set(in_group)):
        raise ValueError("a class appears in more than one tied group")
    units = sorted(groups + [[c] for c in classes if c not in set(in_group)])
    unknown: list[str] = []
    for i in rng.permutation(len(units)):
        if len(unknown) + len(units[i]) <= n_unknown:
            unknown += units[i]
    unknown = sorted(unknown)
    return sorted(set(classes) - set(unknown)), unknown


def near_unknown_split(
    classes: Sequence[str], pairs: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Hand-picked ``near`` split: each unknown class resembles a known one.

    Args:
        classes: Eligible class names.
        pairs: ``{unknown_class: known_class_it_resembles}``. Every key and value must be in
            ``classes`` and no key may also be a value (a class cannot be both).

    Returns:
        ``(known, unknown)``, both sorted; ``unknown`` is exactly the keys of ``pairs``.
    """
    eligible = set(classes)
    unknown = set(pairs)
    for u, k in pairs.items():
        if u not in eligible:
            raise ValueError(f"near split: unknown class {u!r} is not an eligible class")
        if k not in eligible:
            raise ValueError(f"near split: known class {k!r} is not an eligible class")
        if k in unknown:
            raise ValueError(f"near split: {k!r} is listed as both known and unknown")
    if not unknown:
        raise ValueError("near split: pairs is empty")
    known = sorted(eligible - unknown)
    if not known:
        raise ValueError("near split: no known classes remain")
    return known, sorted(unknown)


# ---- Subject folds -------------------------------------------------------------------------
def stratified_assign(
    subjects: Sequence[str],
    members: dict[str, set[str]],
    proportions: Sequence[float],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Deal subjects into bins so that every class's subjects are spread in ``proportions``.

    Iterative stratification for multi-label data (Sechidis, Tsoumakas & Vlahavas 2011), with
    subjects as the examples and "has windows of class c" as the labels.

    Args:
        subjects: The subjects to assign (each exactly once).
        members: ``{class: set of subjects that have it}``; subjects outside ``subjects`` are
            ignored.
        proportions: Share of each bin, summing to 1 (``[1/k] * k`` for k folds,
            ``[1 - val_frac, val_frac]`` for a train/validation separation).
        rng: Decides the order subjects are dealt in and breaks exact ties.

    Returns:
        ``{subject: bin index}``. The class with the fewest unassigned subjects is handled
        first; each of its subjects goes to the bin that still lacks the most subjects of that
        class (ties: the bin that lacks the most subjects overall, then a random one).
        Subjects with no class at all are dealt out last, to the bins lacking the most subjects.
    """
    subjects = sorted(set(subjects))
    props = np.asarray(proportions, dtype=np.float64)
    pool = set(subjects)
    members = {c: set(m) & pool for c, m in members.items()}
    members = {c: m for c, m in members.items() if m}
    classes_of: dict[str, list[str]] = {s: [] for s in subjects}
    for c in sorted(members):
        for s in members[c]:
            classes_of[s].append(c)
    want = {c: len(m) * props for c, m in members.items()}  # subjects of c still wanted per bin
    want_total = len(subjects) * props
    order = [subjects[i] for i in rng.permutation(len(subjects))]
    left = dict.fromkeys(order)  # insertion-ordered set of unassigned subjects
    out: dict[str, int] = {}

    def place(s: str, key: np.ndarray) -> None:
        best = np.flatnonzero(np.isclose(key, key.max()))
        if len(best) > 1:
            t = want_total[best]
            best = best[np.isclose(t, t.max())]
        b = int(best[0] if len(best) == 1 else rng.choice(best))
        out[s] = b
        del left[s]
        want_total[b] -= 1
        for c in classes_of[s]:
            want[c][b] -= 1

    while left:
        remaining = {c: [s for s in left if s in m] for c, m in members.items()}
        remaining = {c: r for c, r in remaining.items() if r}
        if not remaining:
            for s in list(left):
                place(s, want_total)
            break
        c = min(remaining, key=lambda k: (len(remaining[k]), k))
        for s in remaining[c]:
            place(s, want[c])
    return out


def make_subject_folds(
    meta: pd.DataFrame,
    n_folds: int = 5,
    val_frac: float = 0.15,
    seed: int = 0,
    stratify_classes: Sequence[str] | None = None,
    source: str | None = None,
) -> list[dict]:
    """Subject-level folds with a validation subset carved out of each fold's training subjects.

    Args:
        meta: Window metadata with a ``subject`` column (or any frame with one).
        n_folds: Number of folds; each subject is a test subject in exactly one fold.
        val_frac: Fraction of each fold's non-test subjects held out for validation
            (``round(val_frac * n_train)``, at least 1 when there are at least 2 subjects).
        seed: Seed for the subject shuffle and the validation draw (see module docstring).
        stratify_classes: When given, folds and validation subsets are stratified so that the
            subjects of each of these classes are spread evenly (:func:`stratified_assign`;
            needs ``label`` in ``meta``). ``None`` gives the unstratified folds.
        source: With ``stratify_classes``: only windows of this ``source`` count as "the
            subject has the class".

    Returns:
        List of ``n_folds`` dicts with sorted ``train_subjects``, ``val_subjects`` and
        ``test_subjects`` lists; the three are disjoint and together cover every subject.
    """
    subjects = np.array(sorted(meta["subject"].astype(str).unique()), dtype=object)
    if len(subjects) < n_folds:
        raise ValueError(f"{len(subjects)} subjects cannot form {n_folds} folds")
    rng = np.random.default_rng(seed)
    if stratify_classes is not None:
        rows = meta[meta["label"].astype(str).isin(set(stratify_classes))]
        if source is not None:
            rows = rows[rows["source"].astype(str) == source]
        members = {
            str(c): set(g.astype(str))
            for c, g in rows.groupby(rows["label"].astype(str))["subject"]
        }
        fold_of = stratified_assign(subjects.tolist(), members, [1.0 / n_folds] * n_folds, rng)
        folds = []
        for k in range(n_folds):
            pool = sorted(s for s in subjects if fold_of[s] != k)
            part = stratified_assign(pool, members, [1.0 - val_frac, val_frac], rng)
            val = sorted(s for s in pool if part[s] == 1)
            if not val and len(pool) >= 2:  # keep the "at least one" rule of the plain folds
                val = [pool[int(rng.integers(len(pool)))]]
            folds.append(
                {
                    "train_subjects": sorted(set(pool) - set(val)),
                    "val_subjects": val,
                    "test_subjects": sorted(s for s in subjects if fold_of[s] == k),
                }
            )
        return folds
    perm = rng.permutation(len(subjects))
    rank = np.empty(len(subjects), dtype=np.int64)
    rank[perm] = np.arange(len(subjects))  # rank[i] = position of subject i in the shuffle

    folds: list[dict] = []
    gkf = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in gkf.split(np.zeros((len(subjects), 1)), groups=rank):
        pool = sorted(subjects[train_idx].tolist())
        n_val = int(round(val_frac * len(pool))) if len(pool) >= 2 else 0
        n_val = min(max(n_val, 1 if len(pool) >= 2 else 0), len(pool) - 1)
        val = sorted(rng.choice(pool, size=n_val, replace=False).tolist()) if n_val else []
        folds.append(
            {
                "train_subjects": sorted(set(pool) - set(val)),
                "val_subjects": val,
                "test_subjects": sorted(subjects[test_idx].tolist()),
            }
        )
    return folds


# ---- Writing and reading split files -------------------------------------------------------
def write_splits(
    meta: pd.DataFrame,
    out_dir: Path,
    n_folds: int = 5,
    class_split_seeds: Iterable[int] = (0, 1, 2),
    near_pairs: dict[str, str] | None = None,
    min_subjects: int = 20,
    min_minutes: float = 5.0,
    frac_unknown: float = 0.25,
    val_frac: float = 0.15,
    fold_seed: int = 0,
    dataset: str | None = None,
    hop_s: float = 1.0,
    exclude: Iterable[str] = (),
    class_source: str | None = None,
    tied: Iterable[Sequence[str]] = (),
    stratify_folds: bool = False,
) -> list[Path]:
    """Write every ``fold{k}_{classsplit}.json`` for a dataset (spec §3.3).

    Args:
        meta: Window metadata (spec §3.2).
        out_dir: Destination directory, normally ``data/processed/<dataset>/splits``.
        n_folds: Subject folds.
        class_split_seeds: One ``rand{seed}`` class split per seed.
        near_pairs: ``{unknown: resembled_known}`` for the ``near`` split; skipped when None.
        min_subjects, min_minutes: Class eligibility thresholds (see :func:`eligible_classes`).
        frac_unknown: Fraction of eligible classes held out in each random split.
        val_frac: Validation fraction of training subjects per fold.
        fold_seed: Seed for :func:`make_subject_folds`.
        dataset: Dataset name stored in the files; defaults to ``out_dir.parent.name``.
        hop_s: Hop used for the windows, for the minutes count.
        exclude: Labels that are never classes (the non-exercise labels).
        class_source: ``source`` value that class windows are taken from (see the module
            docstring); stored in every split file and honoured by :func:`select`.
        tied: Groups of classes kept on the same side in the random splits
            (see :func:`make_class_split`); the ``near`` split is hand-written and ignores it.
        stratify_folds: Stratify the subject folds over the eligible classes (see the module
            docstring); recorded in each file as ``fold_strategy``.

    Returns:
        Paths of the written files, in ``(fold, class split)`` order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset or out_dir.resolve().parent.name
    classes = eligible_classes(
        meta, min_subjects, min_minutes, hop_s=hop_s, exclude=exclude, source=class_source
    )
    if len(classes) < 2:
        raise ValueError(f"only {len(classes)} eligible classes; relax the thresholds")

    class_splits: list[tuple[str, int | None, list[str], list[str]]] = []
    for s in class_split_seeds:
        known, unknown = make_class_split(
            classes, frac_unknown=frac_unknown, seed=int(s), tied=tied
        )
        class_splits.append((f"rand{int(s)}", int(s), known, unknown))
    if near_pairs is not None:
        known, unknown = near_unknown_split(classes, near_pairs)
        class_splits.append(("near", None, known, unknown))

    folds = make_subject_folds(
        meta,
        n_folds=n_folds,
        val_frac=val_frac,
        seed=fold_seed,
        stratify_classes=classes if stratify_folds else None,
        source=class_source,
    )
    sha = git_sha()
    written: list[Path] = []
    for k, fold in enumerate(folds):
        for name, seed, known, unknown in class_splits:
            split = Split(
                dataset=dataset,
                fold=k,
                n_folds=n_folds,
                class_split=name,
                class_split_seed=seed,
                train_subjects=fold["train_subjects"],
                val_subjects=fold["val_subjects"],
                test_subjects=fold["test_subjects"],
                known_classes=known,
                unknown_classes=unknown,
                eligibility={"min_subjects": int(min_subjects), "min_minutes": float(min_minutes)},
                class_source=class_source,
                fold_strategy="class_stratified" if stratify_folds else "random",
                created_by=sha,
            )
            path = out_dir / f"{split.split_id}.json"
            split.save(path)
            written.append(path)
    return written


def load_split(path: Path) -> Split:
    """Read one ``splits/<split_id>.json`` into a :class:`Split`."""
    return Split.from_dict(json.loads(Path(path).read_text()))


# ---- Selection and encoding ----------------------------------------------------------------
def select(meta: pd.DataFrame, split: Split, part: Part) -> np.ndarray:
    """Window ids of one part of a split.

    Args:
        meta: Window metadata with ``window_id``, ``subject`` and ``label``.
        split: The split.
        part: ``"train"`` / ``"val"``: that fold's subjects, known classes only.
            ``"test_known"``: test subjects, known classes. ``"test_unknown"``: test subjects,
            unknown classes. When ``split.class_source`` is set, only windows with that
            ``source`` are returned.

    Returns:
        int64 array of ``window_id`` values in ``meta`` order.
    """
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    subjects = {
        "train": split.train_subjects,
        "val": split.val_subjects,
        "test_known": split.test_subjects,
        "test_unknown": split.test_subjects,
    }[part]
    classes = split.unknown_classes if part == "test_unknown" else split.known_classes
    mask = meta["subject"].astype(str).isin(subjects) & meta["label"].astype(str).isin(classes)
    if split.class_source is not None:
        mask &= meta["source"].astype(str) == split.class_source
    return meta.loc[mask, "window_id"].to_numpy(dtype=np.int64)


def select_nonexercise(meta: pd.DataFrame, split: Split, labels: list[str]) -> np.ndarray:
    """Window ids of non-exercise windows (P1-b): test subjects, ``source == "multi"``, label in
    ``labels`` (the idle/transition list). Independent of the class split.

    Returns:
        int64 array of ``window_id`` values in ``meta`` order.
    """
    mask = (
        meta["subject"].astype(str).isin(split.test_subjects)
        & (meta["source"].astype(str) == "multi")
        & meta["label"].astype(str).isin(list(labels))
    )
    return meta.loc[mask, "window_id"].to_numpy(dtype=np.int64)


def encode_labels(labels: Sequence[str] | pd.Series | np.ndarray, split: Split) -> np.ndarray:
    """Label strings -> int64 codes: index in ``split.known_classes``; everything else ``-1``.

    Args:
        labels: Any sequence of label strings (shape ``(N,)``).
        split: Supplies the sorted known-class list.

    Returns:
        int64 array of shape ``(N,)``.
    """
    index = split.label_index
    codes = pd.Series(np.asarray(labels, dtype=object)).astype(str).map(index)
    return codes.fillna(-1).to_numpy(dtype=np.int64)
