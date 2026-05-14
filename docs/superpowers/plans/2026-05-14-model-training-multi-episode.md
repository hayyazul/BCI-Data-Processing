# Multi-Episode Model Training Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `model_training.ipynb` so that training and evaluation operate on multiple preprocessed episodes (CSVs under `training_data/`) instead of a single one.

**Architecture:** A flat `episodes: dict[ts, df]` is built once at load time and a parallel `UZ: dict[ts, (U, Z)]` is built once at feature-setup time. Training uses a hardcoded 50/50 slice over sorted timestamps. A new aggregate cell reports per-episode metrics across all loaded episodes. All existing single-episode visualizations are driven by a single `ACTIVE_EPISODE = '...'` string in a dedicated cell that sits *after* training, so episode switches don't require a refit.

**Tech Stack:** Jupyter notebook (`model_training.ipynb`), pandas, numpy, matplotlib, scikit-learn PCA, the existing `LSSM` class (already supports list-of-episodes via `fit(U_list, Z_list)`), and `ArmVisualizer` (unchanged).

**Spec:** `docs/superpowers/specs/2026-05-14-model-training-multi-episode-design.md`.

---

## File Structure

Only one file changes: `model_training.ipynb`. No new Python modules, no changes to `arm_inverse_kinematics.py`, `arm_visualizer.py`, `data_preprocess.ipynb`, or anything in `pose_pipeline/`.

Existing cell layout (from `python claude_scripts/list_ipynb_cells.py model_training.ipynb`):

| Cell idx | Type | Section | Status after this plan |
|----------|------|---------|------------------------|
| 0  | md   | H1 title + stage list | edit (mention multi-episode) |
| 1  | md   | `## 1. Config & imports` | unchanged |
| 2  | code | config + imports | **edit** |
| 3  | md   | `## 2. Load preprocessed training data` | unchanged |
| 4  | code | load CSV | **rewrite** |
| 5  | md   | `## 3. Feature setup` | unchanged |
| 6  | code | build `U`, `Z` | **rewrite** |
| 7  | md   | `## 4. LSSM model definition` | unchanged |
| 8  | code | LSSM class | unchanged |
| 9  | md   | `## 5. Train` | unchanged |
| 10 | code | fit + training MSE print | **rewrite** |
| 11 | md   | `## 6. Evaluation` + `### 6a. Learned dynamics` | unchanged |
| 12 | code | dynamics print | unchanged |
| —  | md   | **NEW** `### 6b. Multi-episode aggregate metrics` | **insert after cell 12** |
| —  | code | **NEW** aggregate table + plots | **insert after the new md** |
| —  | md   | **NEW** `### 6c. Active episode selection` | **insert** |
| —  | code | **NEW** `ACTIVE_EPISODE = ...; ep = get_episode(...)` | **insert** |
| 13 | md   | (was `### 6b. Prediction plots`) | **renumber to `### 6d.`** |
| 14 | code | prediction plots | **edit (read from `ep`)** |
| 15 | md   | (was `### 6c. Cross-correlation`) | **renumber to `### 6e.`** |
| 16 | code | cross-correlation | **edit (read from `ep`)** |
| 17 | md   | (was `### 6d. Inverse PCA → FK animation`) | **renumber to `### 6f.`** |
| 18 | code | joint angle 2D plot | **edit (read from `ep`)** |
| 19 | code | 3D arm animation | **edit (read from `ep`)** |
| 20 | code | empty trailing cell | unchanged |

### How to edit notebook cells

Notebooks are JSON. The reliable approach in this repo is a one-shot Python helper.

Add this helper script (used by every task below):

**Create:** `claude_scripts/edit_ipynb_cell.py`

```python
"""Replace the source of a single cell in a Jupyter notebook by index.

Usage:
    python claude_scripts/edit_ipynb_cell.py <notebook.ipynb> <cell_index> <source_file>

Reads the new cell source from <source_file> (raw text), assigns it to
notebook['cells'][cell_index]['source'] as a list of lines (each line keeping
its trailing newline except possibly the last), clears outputs/execution_count
on code cells, and writes the notebook back in place.
"""
import json
import sys


def main():
    nb_path, idx_str, src_path = sys.argv[1], sys.argv[2], sys.argv[3]
    idx = int(idx_str)
    with open(nb_path) as f:
        nb = json.load(f)
    with open(src_path) as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    cell = nb['cells'][idx]
    cell['source'] = lines
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')


if __name__ == '__main__':
    main()
```

And a helper for inserting new cells (used to add the aggregate + active-episode cells):

**Create:** `claude_scripts/insert_ipynb_cell.py`

```python
"""Insert a new cell at a given index in a Jupyter notebook.

Usage:
    python claude_scripts/insert_ipynb_cell.py <notebook.ipynb> <insert_at_index> <cell_type> <source_file>

cell_type: 'code' or 'markdown'. The cell is inserted at insert_at_index
(existing cells at and after that index shift down by one).
"""
import json
import sys


def main():
    nb_path, idx_str, cell_type, src_path = sys.argv[1:5]
    idx = int(idx_str)
    with open(nb_path) as f:
        nb = json.load(f)
    with open(src_path) as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    cell = {'cell_type': cell_type, 'metadata': {}, 'source': lines}
    if cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    nb['cells'].insert(idx, cell)
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')


if __name__ == '__main__':
    main()
```

Update `claude_scripts/README.md` to list both new scripts.

### How to verify

Notebook work is verified by execution. After each task that modifies a cell, the engineer should run:

```bash
jupyter nbconvert --to notebook --execute model_training.ipynb \
    --output /tmp/model_training_check.ipynb \
    --ExecutePreprocessor.timeout=600
```

If this returns 0 and produces an output notebook, the cells executed without error. Inspect specific outputs via `python claude_scripts/list_ipynb_cells.py /tmp/model_training_check.ipynb --full | sed -n '...,...p'` as needed. **Run All verification is required at the end of every task**; the spec's testing section in `docs/superpowers/specs/2026-05-14-model-training-multi-episode-design.md` enumerates manual sanity checks for the final task.

There are no `pytest`-style tests in this repo — verification is execution + output inspection. That is acceptable here; this is notebook glue code, not library code.

---

## Task 1: Add notebook-cell editing helpers

**Files:**
- Create: `claude_scripts/edit_ipynb_cell.py`
- Create: `claude_scripts/insert_ipynb_cell.py`
- Modify: `claude_scripts/README.md`

- [ ] **Step 1: Create `claude_scripts/edit_ipynb_cell.py`** with the exact contents shown in the "How to edit notebook cells" section above.

- [ ] **Step 2: Create `claude_scripts/insert_ipynb_cell.py`** with the exact contents shown above.

- [ ] **Step 3: Update `claude_scripts/README.md`** to add two bullets:

```markdown
- `edit_ipynb_cell.py` — Replace a single cell's source in a notebook by index. Usage: `python claude_scripts/edit_ipynb_cell.py <nb.ipynb> <idx> <source_file>`. Clears code-cell outputs/execution counts.
- `insert_ipynb_cell.py` — Insert a new cell at a given index. Usage: `python claude_scripts/insert_ipynb_cell.py <nb.ipynb> <insert_at> <code|markdown> <source_file>`.
```

- [ ] **Step 4: Smoke-test on a copy.** Run:

```bash
cp model_training.ipynb /tmp/_smoketest.ipynb
python claude_scripts/list_ipynb_cells.py /tmp/_smoketest.ipynb | head -3
# Read cell 0 source to a file, write it back unchanged, confirm list still works
python -c "import json; nb=json.load(open('/tmp/_smoketest.ipynb')); open('/tmp/_c0.txt','w').write(''.join(nb['cells'][0]['source']))"
python claude_scripts/edit_ipynb_cell.py /tmp/_smoketest.ipynb 0 /tmp/_c0.txt
python claude_scripts/list_ipynb_cells.py /tmp/_smoketest.ipynb | head -3
rm /tmp/_smoketest.ipynb /tmp/_c0.txt
```

Expected: both `list_ipynb_cells.py` invocations print the same `Cell 0 (markdown)` header.

- [ ] **Step 5: Commit**

```bash
git add claude_scripts/edit_ipynb_cell.py claude_scripts/insert_ipynb_cell.py claude_scripts/README.md
git commit -m "Add notebook cell edit/insert helpers"
```

---

## Task 2: Rewrite §1 Config cell (cell 2)

**Files:**
- Modify: `model_training.ipynb` cell 2

- [ ] **Step 1: Write new cell source to `/tmp/cell2.py`:**

```python
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arm_inverse_kinematics import forward_kinematics_fixed, L1_FIXED, L2_FIXED
from arm_visualizer import ArmVisualizer

# ---------------- Config ----------------
TRAINING_DATA_DIR   = 'training_data'
# Which episodes to load. None -> all training_data_*.csv in TRAINING_DATA_DIR.
# Or a list of timestamps, e.g. ['20260425_011156', '20260425_013519'].
EPISODE_TIMESTAMPS  = None

# Feature setup
USE_PCA_EMG         = False    # if True, reduce EMG -> N components
USE_PCA_JOINT       = False    # if True, reduce joints -> N components
N_PCA_EMG           = 2
N_PCA_JOINT         = 2
JOINT_TARGETS       = ['q4']   # which joint columns to predict when not using PCA

# Model
STATE_DIM           = 1
N_RESTARTS          = 10
N_POLISH            = 3
N_BURNIN            = 0
L2_ALPHA            = 1e-5
STANDARDIZE         = True
LEARN_X0            = True
SEED                = 42

# Evaluation
EMG_SAMPLE_RATE_HZ  = 160       # used for FFT freq axis
VIZ_DOWNSAMPLE_N    = 600       # frames for the 3D arm animation
```

- [ ] **Step 2: Apply edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 2 /tmp/cell2.py
```

- [ ] **Step 3: Verify cell 2 contents**

```bash
python claude_scripts/list_ipynb_cells.py model_training.ipynb --preview 200 2>&1 | sed -n '/=== Cell 2/,/=== Cell 3/p'
```

Expected: shows `EPISODE_TIMESTAMPS = None` and *no* mention of `TRAINING_DATA_FILE`.

- [ ] **Step 4: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training §1: replace TRAINING_DATA_FILE with EPISODE_TIMESTAMPS"
```

---

## Task 3: Rewrite §2 Load cell (cell 4)

Builds the `episodes` dict, validates EMG channel consistency, prints a per-episode summary table.

**Files:**
- Modify: `model_training.ipynb` cell 4

- [ ] **Step 1: Write new cell source to `/tmp/cell4.py`:**

```python
# Resolve which timestamps to load.
if EPISODE_TIMESTAMPS is None:
    paths = sorted(glob.glob(os.path.join(TRAINING_DATA_DIR, 'training_data_*.csv')))
    if not paths:
        raise FileNotFoundError(f'No training_data_*.csv files in {TRAINING_DATA_DIR}/')
    timestamps = [os.path.splitext(os.path.basename(p))[0].replace('training_data_', '')
                  for p in paths]
else:
    timestamps = list(EPISODE_TIMESTAMPS)
    paths = [os.path.join(TRAINING_DATA_DIR, f'training_data_{ts}.csv') for ts in timestamps]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f'Missing training_data files: {missing}')

# Load each episode.
episodes = {ts: pd.read_csv(p) for ts, p in zip(timestamps, paths)}

# Derive EMG channel set from the first episode; assert consistency across all.
joint_cols_all = ['q1', 'q2', 'q3', 'q4']
first_ts = timestamps[0]
emg_cols = [c for c in episodes[first_ts].columns
            if c not in ['time'] + joint_cols_all]

for ts in timestamps[1:]:
    cols_here = [c for c in episodes[ts].columns
                 if c not in ['time'] + joint_cols_all]
    if cols_here != emg_cols:
        raise ValueError(
            f'EMG channel mismatch: episode {ts} has {cols_here}, '
            f'expected {emg_cols} (from {first_ts}).'
        )

# Summary table.
summary = pd.DataFrame([
    {
        'timestamp': ts,
        'rows': len(episodes[ts]),
        'duration_s': float(episodes[ts]['time'].iloc[-1] - episodes[ts]['time'].iloc[0]),
        'n_emg_channels': len(emg_cols),
    }
    for ts in timestamps
])
print(f'Loaded {len(episodes)} episodes from {TRAINING_DATA_DIR}/')
print(f'EMG channels  : {emg_cols}')
print(f'Joint columns : {joint_cols_all}')
print()
print(summary.to_string(index=False))
```

- [ ] **Step 2: Apply edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 4 /tmp/cell4.py
```

- [ ] **Step 3: Run cells 0–4 to verify**

```bash
jupyter nbconvert --to notebook --execute model_training.ipynb \
    --output /tmp/check.ipynb \
    --ExecutePreprocessor.timeout=120 \
    2>&1 | tail -30
```

(Note: this also runs later cells which still reference old globals — expect a failure *after* cell 4. The point of this step is that cells 0–4 succeed. The output notebook will be partial.)

To see specifically that cell 4 succeeded:

```bash
python claude_scripts/list_ipynb_cells.py /tmp/check.ipynb --full 2>&1 | sed -n '/=== Cell 4/,/=== Cell 5/p' | head -40
```

Expected: the summary table prints, listing one row per `training_data_*.csv` in `training_data/`.

- [ ] **Step 4: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training §2: load all episodes into dict with channel validation"
```

---

## Task 4: Rewrite §3 Features cell (cell 6)

PCA (if enabled) fits on the pooled frames of all loaded episodes. `build_UZ` produces `(U, Z)` for one episode; we apply it to every loaded episode.

**Files:**
- Modify: `model_training.ipynb` cell 6

- [ ] **Step 1: Write new cell source to `/tmp/cell6.py`:**

```python
from sklearn.decomposition import PCA

# Fit PCA (if enabled) on pooled frames across all loaded episodes.
# PCA is a per-frame linear projection, so pooling frames as samples is
# correct -- this is not temporal stitching.
if USE_PCA_EMG:
    pooled_emg = np.vstack([episodes[ts][emg_cols].values for ts in timestamps])
    pca_emg = PCA(n_components=N_PCA_EMG).fit(pooled_emg)
    print(f'EMG PCA: {pooled_emg.shape[1]} channels -> {N_PCA_EMG} components')
    print(f'  Explained variance ratio: {pca_emg.explained_variance_ratio_}')
    print(f'  Cumulative              : {pca_emg.explained_variance_ratio_.sum():.4f}')
else:
    pca_emg = None
    print(f'EMG PCA disabled.')

if USE_PCA_JOINT:
    pooled_joint = np.vstack([episodes[ts][joint_cols_all].values for ts in timestamps])
    pca_joint = PCA(n_components=N_PCA_JOINT).fit(pooled_joint)
    print(f'Joint PCA: 4 joints -> {N_PCA_JOINT} components')
    print(f'  Explained variance ratio: {pca_joint.explained_variance_ratio_}')
else:
    pca_joint = None
    print(f'Joint PCA disabled. Predicting columns: {JOINT_TARGETS}')


def build_UZ(df):
    """Apply the (optional) PCA transforms + JOINT_TARGETS selection to one episode."""
    emg = df[emg_cols].values
    U = pca_emg.transform(emg) if pca_emg is not None else emg

    joints = df[joint_cols_all].values
    Z = pca_joint.transform(joints) if pca_joint is not None else df[JOINT_TARGETS].values
    return U, Z


# Build (U, Z) once per loaded episode.
UZ = {ts: build_UZ(episodes[ts]) for ts in timestamps}

# Diagnostic print using the first episode.
U0, Z0 = UZ[timestamps[0]]
print(f'\nPer-episode shapes (first episode {timestamps[0]}):')
print(f'  U shape: {U0.shape}  range [{U0.min():.3f}, {U0.max():.3f}]')
print(f'  Z shape: {Z0.shape}  range [{Z0.min():.3f}, {Z0.max():.3f}]')
```

- [ ] **Step 2: Apply edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 6 /tmp/cell6.py
```

- [ ] **Step 3: Verify cells 0–6 execute**

```bash
jupyter nbconvert --to notebook --execute model_training.ipynb \
    --output /tmp/check.ipynb \
    --ExecutePreprocessor.timeout=120 \
    2>&1 | tail -20
```

Inspect cell 6 output:

```bash
python claude_scripts/list_ipynb_cells.py /tmp/check.ipynb --full 2>&1 | sed -n '/=== Cell 6/,/=== Cell 7/p' | head -30
```

Expected: prints "EMG PCA disabled.", "Joint PCA disabled. Predicting columns: ['q4']", and the first-episode shapes. (Later cells will still fail because cell 10 onward references old globals — that's fine for now.)

- [ ] **Step 4: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training §3: build UZ dict and pool-fit PCA across episodes"
```

---

## Task 5: Rewrite §5 Train cell (cell 10)

Hardcoded 50/50 split over sorted timestamps. Fit on the list of training episodes. Special-case for a single loaded episode.

**Files:**
- Modify: `model_training.ipynb` cell 10

- [ ] **Step 1: Write new cell source to `/tmp/cell10.py`:**

```python
# Hardcoded 50/50 split over sorted timestamps.
split = len(timestamps) // 2
train_ts = timestamps[:split]
test_ts  = timestamps[split:]

# Edge case: only one episode loaded -> train on it and reuse it as the test set.
if len(timestamps) == 1:
    print('Only one episode loaded; using it for both training and test.')
    train_ts = list(timestamps)
    test_ts  = list(timestamps)

print(f'Train episodes ({len(train_ts)}): {train_ts}')
print(f'Test  episodes ({len(test_ts)}): {test_ts}')

U_train = [UZ[t][0] for t in train_ts]
Z_train = [UZ[t][1] for t in train_ts]

model = LSSM(state_dim=STATE_DIM,
             input_dim=U_train[0].shape[1],
             output_dim=Z_train[0].shape[1])
model.fit(U_train, Z_train,
          n_restarts=N_RESTARTS,
          n_polish=N_POLISH,
          n_burnin=N_BURNIN,
          alpha=L2_ALPHA,
          learn_x0=LEARN_X0,
          standardize=STANDARDIZE,
          seed=SEED)

# Report training MSE pooled across training episodes.
sse = 0.0
count = 0
for U_e, Z_e in zip(U_train, Z_train):
    Z_hat_e = model.predict(U_e)
    sse   += np.sum((Z_hat_e - Z_e) ** 2)
    count += Z_e.size
print(f'\nTraining MSE (pooled over train episodes): {sse / count:.6f}')
```

- [ ] **Step 2: Apply edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 10 /tmp/cell10.py
```

- [ ] **Step 3: Verify cells 0–10 execute and §6a still runs**

```bash
jupyter nbconvert --to notebook --execute model_training.ipynb \
    --output /tmp/check.ipynb \
    --ExecutePreprocessor.timeout=600 \
    2>&1 | tail -30
```

This will run all cells but later ones (14, 16, 18, 19) still reference old globals (`U`, `Z`, `Z_hat`, `full_joint_data`, `df_model`) and will fail. The point of this step is that cells 0–12 succeed. Inspect cell 10 output:

```bash
python claude_scripts/list_ipynb_cells.py /tmp/check.ipynb --full 2>&1 | sed -n '/=== Cell 10/,/=== Cell 11/p' | head -40
```

Expected: prints train/test episode lists, L-BFGS-B polish output, and a finite training MSE.

- [ ] **Step 4: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training §5: train on list of episodes via 50/50 split"
```

---

## Task 6: Insert §6b multi-episode aggregate cell (new markdown + code, between cells 12 and 13)

After this task the cell indices for everything originally at index ≥ 13 shift by +2.

**Files:**
- Modify: `model_training.ipynb` (insert two new cells at indices 13 and 14)

- [ ] **Step 1: Write the markdown source to `/tmp/cell_md_6b.md`:**

```markdown
### 6b. Multi-episode aggregate metrics

Per-episode MSE / R² / Pearson r over every loaded episode, with a `train` flag distinguishing training episodes from held-out ones. Followed by a bar chart of per-episode MSE and a pooled scatter of predicted vs. true (one subplot per target, colored by episode).
```

- [ ] **Step 2: Insert the markdown cell at index 13**

```bash
python claude_scripts/insert_ipynb_cell.py model_training.ipynb 13 markdown /tmp/cell_md_6b.md
```

- [ ] **Step 3: Write the code source to `/tmp/cell_code_6b.py`:**

```python
from scipy.stats import pearsonr

# Per-episode metrics.
rows = []
preds_per_ts = {}    # ts -> (Z, Z_hat) for the scatter below
for ts in timestamps:
    U_e, Z_e = UZ[ts]
    Z_hat_e  = model.predict(U_e)
    preds_per_ts[ts] = (Z_e, Z_hat_e)

    row = {'timestamp': ts, 'train': ts in train_ts}
    n_targets = Z_e.shape[1]
    for j in range(n_targets):
        target_name = JOINT_TARGETS[j] if (pca_joint is None and j < len(JOINT_TARGETS)) else f'Z{j+1}'
        err = Z_hat_e[:, j] - Z_e[:, j]
        mse = float(np.mean(err ** 2))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((Z_e[:, j] - Z_e[:, j].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        r  = float(pearsonr(Z_e[:, j], Z_hat_e[:, j])[0]) if Z_e.shape[0] > 1 else float('nan')
        row[f'mse_{target_name}']  = mse
        row[f'r2_{target_name}']   = r2
        row[f'pear_{target_name}'] = r
    rows.append(row)

metrics_df = pd.DataFrame(rows)
print(metrics_df.to_string(index=False))

# --- Bar chart: per-episode MSE per target, colored by train vs. test ---
target_names = [c[len('mse_'):] for c in metrics_df.columns if c.startswith('mse_')]
n_targets    = len(target_names)
plt.close('all')
fig, axes = plt.subplots(n_targets, 1, figsize=(12, 2.5 * n_targets), sharex=True)
if n_targets == 1:
    axes = [axes]
x = np.arange(len(metrics_df))
colors = ['tab:blue' if t else 'tab:orange' for t in metrics_df['train']]
for j, tname in enumerate(target_names):
    axes[j].bar(x, metrics_df[f'mse_{tname}'], color=colors)
    axes[j].set_ylabel(f'MSE  {tname}')
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(metrics_df['timestamp'], rotation=45, ha='right')
# Manual legend.
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(facecolor='tab:blue',   label='train'),
                        Patch(facecolor='tab:orange', label='test')],
               loc='upper right')
plt.suptitle('Per-episode MSE')
plt.tight_layout()
plt.show()

# --- Pooled scatter: predicted vs. true, colored by episode, one subplot per target ---
fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5), squeeze=False)
axes = axes[0]
cmap = plt.get_cmap('tab20')
for j, tname in enumerate(target_names):
    ax = axes[j]
    all_z = []
    for k, ts in enumerate(timestamps):
        Z_e, Z_hat_e = preds_per_ts[ts]
        ax.scatter(Z_e[:, j], Z_hat_e[:, j], s=4, alpha=0.4,
                   color=cmap(k % 20), label=ts)
        all_z.append(Z_e[:, j]); all_z.append(Z_hat_e[:, j])
    lo, hi = float(np.min(np.concatenate(all_z))), float(np.max(np.concatenate(all_z)))
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel(f'true {tname}')
    ax.set_ylabel(f'pred {tname}')
    ax.set_title(tname)
axes[-1].legend(fontsize=6, loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.suptitle('Predicted vs true, pooled across episodes')
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Insert the code cell at index 14**

```bash
python claude_scripts/insert_ipynb_cell.py model_training.ipynb 14 code /tmp/cell_code_6b.py
```

- [ ] **Step 5: Verify the two cells exist and render**

```bash
python claude_scripts/list_ipynb_cells.py model_training.ipynb --preview 200 2>&1 | sed -n '/=== Cell 13/,/=== Cell 15/p'
```

Expected: cell 13 is the new markdown, cell 14 is the new code, cell 15 is the old `### 6b. Prediction plots` markdown (now to be renumbered in Task 8).

Execute the notebook through this point:

```bash
jupyter nbconvert --to notebook --execute model_training.ipynb \
    --output /tmp/check.ipynb \
    --ExecutePreprocessor.timeout=600 \
    2>&1 | tail -20
```

The new aggregate cell should succeed. Cells after 14 (the old viz cells) will still fail — that's expected until Task 8.

- [ ] **Step 6: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training §6b: add multi-episode aggregate metrics + plots"
```

---

## Task 7: Insert §6c active-episode cell (new markdown + code, at indices 15 and 16)

After Task 6, the cell indices look like:

| idx | content |
|-----|---------|
| 13 | (new) §6b markdown |
| 14 | (new) §6b code |
| 15 | old `### 6b. Prediction plots` markdown — to be renumbered in Task 8 |
| 16 | old prediction-plots code |
| 17 | old `### 6c. Cross-correlation` markdown |
| ... | ... |

Insert the new §6c markdown + code at indices 15 and 16 (pushing the old §6b/prediction-plots markdown to 17, etc.).

**Files:**
- Modify: `model_training.ipynb`

- [ ] **Step 1: Write the markdown source to `/tmp/cell_md_6c.md`:**

```markdown
### 6c. Active episode selection

The visualizations below (prediction plots, cross-correlation, joint plot, 3D arm animation) all read from `ep`. Edit `ACTIVE_EPISODE` in this cell and rerun this cell + the cells below to switch episodes — no model refit needed.
```

- [ ] **Step 2: Insert the markdown cell at index 15**

```bash
python claude_scripts/insert_ipynb_cell.py model_training.ipynb 15 markdown /tmp/cell_md_6c.md
```

- [ ] **Step 3: Write the code source to `/tmp/cell_code_6c.py`:**

```python
from types import SimpleNamespace

ACTIVE_EPISODE = train_ts[0]   # <-- edit this one line to switch episodes


def get_episode(ts):
    """Bundle everything the §6d/6e/6f cells need for one episode."""
    if ts not in episodes:
        raise KeyError(f'{ts!r} not in loaded episodes: {list(episodes.keys())}')
    df = episodes[ts]
    U, Z = UZ[ts]
    Z_hat = model.predict(U)
    return SimpleNamespace(
        ts=ts,
        df=df,
        U=U,
        Z=Z,
        Z_hat=Z_hat,
        full_joint_data=df[joint_cols_all].values,
        is_train=(ts in train_ts),
    )


ep = get_episode(ACTIVE_EPISODE)
print(f'Active episode: {ep.ts}  (train={ep.is_train})  '
      f'U shape {ep.U.shape}  Z shape {ep.Z.shape}')
```

- [ ] **Step 4: Insert the code cell at index 16**

```bash
python claude_scripts/insert_ipynb_cell.py model_training.ipynb 16 code /tmp/cell_code_6c.py
```

- [ ] **Step 5: Verify**

```bash
python claude_scripts/list_ipynb_cells.py model_training.ipynb --preview 200 2>&1 | sed -n '/=== Cell 15/,/=== Cell 18/p'
```

Expected: cell 15 = new §6c markdown, cell 16 = new §6c code, cell 17 = old `### 6b. Prediction plots` markdown (still to renumber), cell 18 = old prediction-plots code (still to update to read `ep`).

- [ ] **Step 6: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training §6c: add active-episode selection cell"
```

---

## Task 8: Update §6d prediction-plots cells (markdown at 17, code at 18)

Renumber the markdown header `6b -> 6d`. Update the code cell to read from `ep`.

**Files:**
- Modify: `model_training.ipynb` cells 17 and 18

- [ ] **Step 1: Write the renumbered markdown to `/tmp/cell_md_6d.md`:**

```markdown
### 6d. Prediction plots — true vs predicted targets
```

- [ ] **Step 2: Apply markdown edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 17 /tmp/cell_md_6d.md
```

- [ ] **Step 3: Write the updated code to `/tmp/cell_code_6d.py`:**

```python
plt.close('all')
n_out = ep.Z.shape[1]
fig, ax = plt.subplots(n_out, 1, figsize=(12, 2.5*n_out), sharex=True)
if n_out == 1:
    ax = [ax]
for i in range(n_out):
    ax[i].plot(ep.Z[:, i],     label=f'true  Z{i+1}', alpha=0.8)
    ax[i].plot(ep.Z_hat[:, i], label=f'pred Z{i+1}', alpha=0.8)
    ax[i].set_ylabel(f'Z{i+1}')
    ax[i].legend(loc='upper right')
ax[-1].set_xlabel('Time (samples)')
plt.suptitle(f'True vs Predicted — episode {ep.ts} (train={ep.is_train})')
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Apply code edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 18 /tmp/cell_code_6d.py
```

- [ ] **Step 5: Verify**

```bash
python claude_scripts/list_ipynb_cells.py model_training.ipynb --preview 200 2>&1 | sed -n '/=== Cell 17/,/=== Cell 19/p'
```

Expected: cell 17 header reads `### 6d.`, cell 18 references `ep.Z`, `ep.Z_hat`, `ep.ts`, `ep.is_train`.

- [ ] **Step 6: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training §6d: prediction plots read from active episode"
```

---

## Task 9: Update §6e cross-correlation cells (markdown at 19, code at 20)

**Files:**
- Modify: `model_training.ipynb` cells 19 and 20

- [ ] **Step 1: Write the renumbered markdown to `/tmp/cell_md_6e.md`:**

```markdown
### 6e. Input/output cross-correlation (lag analysis)
```

- [ ] **Step 2: Apply markdown edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 19 /tmp/cell_md_6e.md
```

- [ ] **Step 3: Write the updated code to `/tmp/cell_code_6e.py`:**

```python
from scipy.signal import correlate
from scipy.stats import pearsonr

U_a = ep.U
Z_a = ep.Z

print(f'Active episode: {ep.ts}')
print('Pearson correlations (per-channel):')
for i in range(min(U_a.shape[1], 4)):
    for j in range(Z_a.shape[1]):
        r, p = pearsonr(U_a[:, i], Z_a[:, j])
        print(f'  U{i+1} vs Z{j+1}: r={r:+.4f}  (p={p:.4f})')

print('\nCross-correlation peak lags:')
for i in range(min(U_a.shape[1], 4)):
    for j in range(Z_a.shape[1]):
        x = U_a[:, i] - U_a[:, i].mean()
        y = Z_a[:, j] - Z_a[:, j].mean()
        c = correlate(y, x, mode='full') / (np.std(x)*np.std(y)*len(x))
        lags = np.arange(-len(x)+1, len(x))
        peak_idx = np.argmax(np.abs(c))
        print(f'  U{i+1}->Z{j+1}: peak corr {c[peak_idx]:+.3f} at lag {lags[peak_idx]} samples')
```

- [ ] **Step 4: Apply code edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 20 /tmp/cell_code_6e.py
```

- [ ] **Step 5: Verify**

```bash
python claude_scripts/list_ipynb_cells.py model_training.ipynb --preview 200 2>&1 | sed -n '/=== Cell 19/,/=== Cell 21/p'
```

Expected: cell 19 header reads `### 6e.`, cell 20 references `ep.U`, `ep.Z`, `ep.ts`.

- [ ] **Step 6: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training §6e: cross-correlation reads from active episode"
```

---

## Task 10: Update §6f joint-plot + 3D-animation cells (markdown at 21, code at 22 and 23)

**Files:**
- Modify: `model_training.ipynb` cells 21, 22, and 23

- [ ] **Step 1: Write the renumbered markdown to `/tmp/cell_md_6f.md`:**

```markdown
### 6f. Inverse PCA -> joint angles, then FK arm animation

If joint PCA was used, invert it back to (q1..q4). Otherwise we predicted joint columns directly; the remaining joints are filled in from ground truth so the FK animation has all four angles. Drives off the active episode.
```

- [ ] **Step 2: Apply markdown edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 21 /tmp/cell_md_6f.md
```

- [ ] **Step 3: Write the joint-plot code to `/tmp/cell_code_6f_a.py`:**

```python
# Reconstruct full (q1..q4) from predictions for the active episode.
if pca_joint is not None:
    q_hat_full = pca_joint.inverse_transform(ep.Z_hat)
else:
    # Predictions only cover JOINT_TARGETS; copy ground truth for the rest.
    q_hat_full = ep.full_joint_data.copy()
    for k, col in enumerate(JOINT_TARGETS):
        col_idx = joint_cols_all.index(col)
        q_hat_full[:, col_idx] = ep.Z_hat[:, k]

q_gt_full = ep.full_joint_data

fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for i, col in enumerate(joint_cols_all):
    ax[i].plot(q_gt_full[:, i],  label='ground truth', alpha=0.8)
    ax[i].plot(q_hat_full[:, i], label='predicted',    alpha=0.8)
    ax[i].set_ylabel(col)
ax[0].legend()
ax[-1].set_xlabel('time (samples)')
plt.suptitle(f'Joint angles: True vs Predicted — episode {ep.ts} (train={ep.is_train})')
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Apply joint-plot edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 22 /tmp/cell_code_6f_a.py
```

- [ ] **Step 5: Write the 3D-animation code to `/tmp/cell_code_6f_b.py`:**

```python
# 3D arm animation: ground truth vs EMG-reconstructed (active episode).
T_full = q_hat_full.shape[0]
T_viz  = min(VIZ_DOWNSAMPLE_N, T_full)
step   = max(1, T_full // T_viz)
idx    = np.arange(0, T_full, step)[:T_viz]

q_gt_ds  = q_gt_full[idx]
q_hat_ds = q_hat_full[idx]
times    = ep.df['time'].values[idx]

viz = ArmVisualizer(fk_func=forward_kinematics_fixed, L1=L1_FIXED, L2=L2_FIXED)
viz.add_arm(name=f'Ground truth ({ep.ts})', color='green', dash='dash',
            q1=q_gt_ds[:, 0], q2=q_gt_ds[:, 1], q3=q_gt_ds[:, 2], q4=q_gt_ds[:, 3])
viz.add_arm(name=f'EMG-Reconstructed ({ep.ts})', color='yellow',
            q1=q_hat_ds[:, 0], q2=q_hat_ds[:, 1], q3=q_hat_ds[:, 2], q4=q_hat_ds[:, 3])
viz.show(times=times)
```

- [ ] **Step 6: Apply 3D-animation edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 23 /tmp/cell_code_6f_b.py
```

- [ ] **Step 7: Verify**

```bash
python claude_scripts/list_ipynb_cells.py model_training.ipynb --preview 200 2>&1 | sed -n '/=== Cell 21/,/=== Cell 24/p'
```

Expected: cell 21 markdown reads `### 6f.`, cells 22 and 23 reference `ep.Z_hat`, `ep.full_joint_data`, `ep.df`, `ep.ts`.

- [ ] **Step 8: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training §6f: joint plot + 3D arm animation read from active episode"
```

---

## Task 11: Update top-level markdown cell 0 (stage list)

**Files:**
- Modify: `model_training.ipynb` cell 0

- [ ] **Step 1: Write new cell 0 source to `/tmp/cell0.md`:**

```markdown
# Model Training & Evaluation

Train a Numba-jitted Linear State-Space Model (LSSM) that maps EMG -> joint angles using one or more preprocessed `df_model` files written by `data_preprocess.ipynb`.

Stages:
1. Config + imports
2. Load preprocessed training data (one or more episodes)
3. Feature setup (optional PCA on EMG / joints, fit pooled across episodes)
4. LSSM model definition
5. Train (hardcoded 50/50 split over loaded episodes)
6. Evaluation:
   - 6a. Learned dynamics
   - 6b. Multi-episode aggregate metrics across loaded episodes
   - 6c. Active-episode selection (one-line switch for the cells below)
   - 6d. Prediction plots (active episode)
   - 6e. Cross-correlation (active episode)
   - 6f. Joint plot + 3D arm animation (active episode)
```

- [ ] **Step 2: Apply edit**

```bash
python claude_scripts/edit_ipynb_cell.py model_training.ipynb 0 /tmp/cell0.md
```

- [ ] **Step 3: Commit**

```bash
git add model_training.ipynb
git commit -m "model_training: update overview to reflect multi-episode flow"
```

---

## Task 12: Full Run All verification

This is the spec's testing section, executed end-to-end.

**Files:**
- None modified.

- [ ] **Step 1: Run All from a clean state with `EPISODE_TIMESTAMPS = None`**

```bash
jupyter nbconvert --to notebook --execute model_training.ipynb \
    --output model_training.ipynb \
    --ExecutePreprocessor.timeout=1800
```

Expected: exit code 0; notebook contains rendered outputs for every cell.

- [ ] **Step 2: Inspect §2 summary table**

```bash
python claude_scripts/list_ipynb_cells.py model_training.ipynb --full 2>&1 | sed -n '/=== Cell 4/,/=== Cell 5/p' | head -40
```

Expected: one row per `training_data_*.csv` in `training_data/` (currently 17 files), all with the same EMG channel count.

- [ ] **Step 3: Inspect §5 train/test split**

```bash
python claude_scripts/list_ipynb_cells.py model_training.ipynb --full 2>&1 | sed -n '/=== Cell 12/,/=== Cell 13/p' | head -20
```

Expected: `Train episodes` list has roughly half the loaded timestamps; `Test episodes` has the other half.

- [ ] **Step 4: Inspect §6b aggregate table**

```bash
python claude_scripts/list_ipynb_cells.py model_training.ipynb --full 2>&1 | sed -n '/=== Cell 14/,/=== Cell 15/p' | head -40
```

Expected: one row per loaded episode with a `train` boolean column and finite numeric metrics; bar chart and pooled scatter render without error (visible by opening the notebook in Jupyter, not by `nbconvert`).

- [ ] **Step 5: Switch active episode to a test-set timestamp and rerun §6c–§6f only**

Open the notebook in Jupyter, change the `ACTIVE_EPISODE` line in cell 16 to a timestamp from `test_ts` (e.g. one of the latter half of the sorted list), and run cells 16, 18, 20, 22, 23. Confirm:
- §6c prints `train=False`.
- Prediction plots, joint plot, and 3D animation refresh with the new episode's data.
- No model refit happened (cell 12 was not rerun).

(This step is interactive — there is no nbconvert equivalent for "rerun a subset." Document the outcome in the commit message.)

- [ ] **Step 6: Single-episode sanity check**

In a separate scratch run, set `EPISODE_TIMESTAMPS = ['20260425_013519']` in cell 2, Run All, confirm:
- §2 reports 1 episode.
- §5 prints "Only one episode loaded; using it for both training and test."
- §6b table has one row with `train=True`.
- §6d/6e/6f render normally.

Restore `EPISODE_TIMESTAMPS = None` afterwards.

- [ ] **Step 7: Commit the executed notebook**

```bash
git add model_training.ipynb
git commit -m "model_training: run-all verification with all loaded episodes"
```

---

## Self-Review Notes

- **Spec coverage:** every requirement in the spec maps to a task:
  - "Load subset of episodes by timestamp" → Tasks 2, 3.
  - "Configurable prediction cell on any loaded data" → Task 7 (active episode), Task 8 (prediction cell).
  - "Applies to other viz cells" → Tasks 9, 10.
  - "One-line change to switch episode across multiple cells" → Task 7 (`ACTIVE_EPISODE = '...'`).
  - "Multi-episode test results visualized" → Task 6 (aggregate cell).
  - EMG channel mismatch → Task 3.
  - PCA fit on pooled frames → Task 4.
  - Single-episode special case → Task 5.
- **Type consistency:** `ep` is a `SimpleNamespace` with attrs `ts`, `df`, `U`, `Z`, `Z_hat`, `full_joint_data`, `is_train` — referenced consistently in Tasks 8, 9, 10. `train_ts`/`test_ts` lists defined in Task 5 and used in Tasks 6, 7. `pca_joint` is `None` when off — code in Task 10 checks `pca_joint is not None` (matches `build_UZ` in Task 4).
- **Placeholder scan:** no TBDs, no "similar to" references, every code step has actual code.
- **Cell index drift:** Tasks 6 and 7 each insert two cells; subsequent tasks reference the post-insert indices explicitly in their step text.
