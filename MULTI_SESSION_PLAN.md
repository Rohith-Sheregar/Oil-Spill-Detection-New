# Multi-Session Kaggle Training Guide
## Oil Spill Detection — Module 1

> **Problem:** Kaggle free sessions are limited to 12 hours and ~30 GPU-hours/week.  
> **Solution:** Run 20 epochs per session, save `last_model.pt` to a Kaggle Dataset, resume in the next session.

---

## 📐 Training Budget Estimate (T4 GPU)

| Epochs | Est. Time | Sessions Needed |
|--------|-----------|-----------------|
| 1–20   | ~2–3 hrs  | Session 1       |
| 21–40  | ~2–3 hrs  | Session 2       |
| 41–50  | ~1.5 hrs  | Session 3       |
| Pseudo-labels (+5 cycles) | ~1 hr | Session 3 (final) |

**Total: ~50 supervised epochs + 5 pseudo cycles across 3 sessions.**

---

## 🗺️ Session-by-Session Workflow

### Session 1 — Fresh Start (Epochs 1–20)

**In the Kaggle notebook UI:**
1. Open `New/module-1-training.ipynb` on Kaggle
2. Enable GPU: **Settings (⚙️) → Accelerator → GPU T4 x1 → Save**
3. In **Cell 3**: leave `CHECKPOINT_DATASET = ""`  
4. In **Cell 4**: set `EPOCHS_THIS_SESSION = 20`, `PSEUDO_CYCLES = 0`
5. Run all cells (Cell 0 → Cell 5)
6. After Cell 5 finishes, go to **Output tab → Download**:
   - `best_model.pt`
   - `last_model.pt`
   - `train_metrics.csv`

**After the session:**
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"New Dataset"** → name it `oil-spill-checkpoints`
3. Upload `best_model.pt`, `last_model.pt`, `train_metrics.csv`
4. Set visibility to **Private**
5. Note the dataset slug (e.g., `yourusername/oil-spill-checkpoints`)

---

### Session 2 — Resume (Epochs 21–40)

**Before starting:**
1. In the Kaggle notebook, click **"Add Data"** → search for `oil-spill-checkpoints`
2. Add your private dataset — it will mount at `/kaggle/input/oil-spill-checkpoints/`

**In the notebook:**
1. In **Cell 3**: set `CHECKPOINT_DATASET = "oil-spill-checkpoints"`
2. In **Cell 4**: set `EPOCHS_THIS_SESSION = 20`, `PSEUDO_CYCLES = 0`
3. Run all cells

The script will automatically:
- Load `last_model.pt` from the dataset
- Continue training from epoch 21
- Append new rows to `train_metrics.csv`
- Update `best_model.pt` if a better checkpoint is found

**After the session:** Upload new `.pt` files to the same Kaggle Dataset (update the existing version).

---

### Session 3 — Final Session (Epochs 41–50 + Pseudo-Labels)

**In the notebook:**
1. In **Cell 3**: `CHECKPOINT_DATASET = "oil-spill-checkpoints"`
2. In **Cell 4**: set `EPOCHS_THIS_SESSION = 10`, `PSEUDO_CYCLES = 5`
3. Run all cells

The training script will:
- Finish the last 10 supervised epochs
- Run 5 pseudo-label self-evolution cycles on lookalike/no-oil data
- Generate the HTML report at `/kaggle/working/results/module1/report/module1_report.html`

**Download everything from the Output tab.**

---

## 🔧 Kaggle Dataset — Update Workflow (Session 2+)

```
Kaggle Dataset: oil-spill-checkpoints
  ├── best_model.pt    ← overwrite with latest
  ├── last_model.pt    ← overwrite with latest
  └── train_metrics.csv ← overwrite with cumulative file
```

To update a dataset version:
1. Go to your dataset page on Kaggle
2. Click **"New Version"** → drag-and-drop the new files
3. Wait for the version to be processed (usually < 5 min)
4. In your next notebook session, the updated files are automatically available

---

## 🚨 Important Notes

### GPU Must Be Enabled Manually
The notebook metadata requests a T4 GPU, but Kaggle may not honour this automatically when you re-open the notebook. **Always verify Cell 0 prints the GPU name before running Cell 4.**

### Weekly GPU Quota
Kaggle free accounts get ~30 GPU-hours/week. If you hit the quota:
- Switch from T4 to **CPU** (much slower, not recommended for training)
- Wait until the next week's quota resets
- Consider using Kaggle's P100 GPU (same quota, ~2× faster)

### Choosing Which Checkpoint to Resume From

| Checkpoint | Use Case |
|------------|----------|
| `last_model.pt` | Normal session continuation (recommended) |
| `best_model.pt` | Fine-tuning from the best weights (use for pseudo-label sessions) |

In Cell 3, set `PREFER_CHECKPOINT = "last_model.pt"` (default) for normal resumption.

### Epoch Counting
The `--epochs N` flag means "run N more epochs **starting from where you left off**". So if you stopped at epoch 20 and pass `--epochs 20`, the script trains epochs 21–40.

---

## 📋 Quick Reference Checklist

### Start of Every Session
- [ ] GPU enabled in Settings (Cell 0 confirms it)
- [ ] Repo cloned / pulled latest (Cell 1)
- [ ] Data symlinks created (Cell 2)
- [ ] Correct checkpoint attached + `CHECKPOINT_DATASET` set (Cell 3)

### End of Every Session
- [ ] Cell 5 ran successfully
- [ ] `best_model.pt` and `last_model.pt` downloaded
- [ ] Uploaded to Kaggle Dataset `oil-spill-checkpoints`
- [ ] Dataset version updated and processed

### Final Session Only
- [ ] `PSEUDO_CYCLES = 5` set in Cell 4
- [ ] HTML report downloaded from Output tab
- [ ] All metrics CSV downloaded for analysis
