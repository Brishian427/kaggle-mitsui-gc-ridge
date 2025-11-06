# Output Directory

This directory contains generated output files from the training pipeline.

## Generated Files

After running the training pipeline, the following files will be created:

- `val_y_true.npy` - Validation ground truth (n_days, 424)
- `val_y_pred.npy` - Raw validation predictions (n_days, 424)
- `val_y_pred_hard.npy` - Hard neutralized predictions (0.42+ Sharpe)
- `val_y_pred_shrink.npy` - Shrink neutralized predictions (alternative)
- `val_destroyers_idx.npy` - Destroyer target indices

## Usage

These files are generated automatically when you run:

```bash
python run_pipeline.py
```

Or run the steps individually:

```bash
python src/train_model.py          # Generates val_y_true.npy and val_y_pred.npy
python src/neutralize_destroyers.py  # Generates remaining files
```

## Notes

- This directory is automatically created by the pipeline
- Output files are typically large (several MB each)
- Files are saved in NumPy binary format (.npy)
- This directory is ignored by git (see .gitignore)

