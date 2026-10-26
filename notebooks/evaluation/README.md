# W&B model evaluation

This directory contains a standalone evaluator for checkpoints produced by the
multilingual MV-CLIP code in `notebooks/3rd_exp`. It downloads a run's history
and best model artifact, recreates the test split, runs inference, and writes:

- `training_history.csv` and `training_history.png`
- `predictions.csv` (image ID, label, prediction, and class probabilities)
- `metrics.json`
- `confusion_matrix.png`
- `probability_distribution.png`

## Run

From this directory (or pass the same arguments from the repository root):

```bash
uv run evaluate.py \
  --run-path ENTITY/skripsi_3rd-exp/RUN_ID \
  --variant mean_mean
```

Use `--variant mean_cls` for a checkpoint trained by
`mv_clip_multilingual_mean_cls.py`. W&B must already be authenticated; if it is
not, run `wandb login` or set `WANDB_API_KEY`.

The default dataset location is auto-detected as
`datasets/MMSD2.0/data`. Override it when needed:

```bash
uv run evaluate.py --run-path ENTITY/PROJECT/RUN_ID \
  --dataset-dir /path/to/MMSD2.0/data \
  --output-dir results/my-run \
  --batch-size 16
```

By default the evaluator chooses the newest model artifact logged by the run
and requests its `best` alias. Use `--artifact-name NAME:VERSION_OR_ALIAS` to
select one explicitly. `--device auto` chooses CUDA when available.

