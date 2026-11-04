# Data pipeline

How to get from downloads to windows and splits, what the numbers are, and where they come from.
Commands are copy-pasteable and re-run before each milestone.

## Commands

```bash
uv run scripts/prepare_data.py recofit --steps recordings windows splits
uv run scripts/prepare_data.py recofit --steps labels        # per-label minutes and subjects
uv run scripts/prepare_data.py wearable --steps recordings windows
```

Inputs live under `data/external/<dataset>/` (gitignored); outputs under `data/processed/<dataset>/`
in the layout of [design/development-spec.md §3](design/development-spec.md#3-data-contracts).

## RecoFit

Source files (copied 2026-11-03 from the FormCoach machine; same public LFS objects):

| File | Bytes | SHA-256 |
|---|---|---|
| `exercise_data.50.0000_singleonly.mat` | 1,562,300,284 | (filled by `recordings.meta.json`) |
| `exercise_data.50.0000_multionly.mat` | 1,571,881,721 | (filled by `recordings.meta.json`) |

Label list, counts, window counts and the eligibility decision: filled in when the first
`recordings` and `windows` steps finish (item 4.3, 4.4).

## Our wearable

See [wearable-recording.md](wearable-recording.md) for the sessions. The `wearable` dataset is
built from `data/wearable/<subject>/<session>/` by `uaar.data.wearable.to_recordings`.
