# Final models

The final copy of every model the live page or the report depends on. `reports/runs/` holds
all runs of all sweeps and is not in git; a model is copied here with
`uv run scripts/export_model.py <run_dir>`, and `scripts/finetune.py` writes here directly.
Each folder loads like a run directory (`uaar.models.factory.load`, `configs/demo.yaml`).

| Folder | What it is |
|---|---|
| `k4plain_recofit_{softmax,mcdropout,edl,logreg}_fold0_bg0_seed0` | the four methods trained on the four exercises only (fold 0, background split `bg0`) |
| `k4taught_recofit_{softmax,mcdropout,edl}_fold0_bg0_seed0` | the taught models: also trained to be uncertain on other RecoFit movements |
| `k4other_recofit_softmax_fold0_bg0_seed0` | the reference softmax with an explicit "Other" class |
| `*_ft-<subject>` | a model above fine-tuned on that person's wearable recordings (`scripts/finetune.py`); `finetune.json` and `report.md` say how and how well |

Fold 0 is one of five folds; the numbers in the report are means over all folds and background
splits, not the numbers of these particular models.
