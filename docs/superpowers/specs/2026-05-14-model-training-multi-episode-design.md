# Multi-Episode Support for `model_training.ipynb`

**Date:** 2026-05-14
**Scope:** `model_training.ipynb` only. `data_preprocess.ipynb` already emits one `training_data_{timestamp}.csv` per session — no changes needed there.

## Goal

Train and evaluate the LSSM on multiple episodes at once. Allow the user to:

1. Load a chosen subset of preprocessed episodes by timestamp.
2. Train on a designated subset of those (default: a hardcoded 50/50 slice).
3. View aggregate test results across all loaded episodes.
4. Drill into any one loaded episode for single-episode visualizations (prediction plots, cross-correlation, joint plot, 3D arm animation) by editing a single line.

The existing `LSSM.fit(U, Z, ...)` already accepts a list of per-episode `(U_i, Z_i)` arrays, so the model side is unchanged. The work is in the notebook cells that build inputs, dispatch training, and visualize.

## Non-goals

- No automatic train/test split selection — the split is a hardcoded 50/50 slice over sorted timestamps, edited in place if desired.
- No persistence of loaded episodes to disk — rebuilt from CSVs on each notebook run.
- No support for episodes whose EMG channel sets differ from each other — channel mismatch is a hard error at load time. (Channel-set drift is a preprocessing bug, not a modeling concern.)
- No changes to `data_preprocess.ipynb`, `LSSM`, `ArmVisualizer`, or `arm_inverse_kinematics.py`.

## Data model

A flat dict keyed by timestamp string:

```python
episodes: dict[str, pd.DataFrame]   # ts -> df_model for that session
UZ:       dict[str, tuple[np.ndarray, np.ndarray]]   # ts -> (U, Z)
```

`emg_cols` and `joint_cols_all` are derived once from the first loaded episode and asserted identical across all others.

## Cell-by-cell plan

| Cell | Purpose | Status |
|------|---------|--------|
| §1 Config & imports | Add `EPISODE_TIMESTAMPS = None`. Remove `TRAINING_DATA_FILE`. Keep other knobs. | edit |
| §2 Load | Build `episodes` dict from `EPISODE_TIMESTAMPS` (or glob all if `None`). Validate EMG channel consistency. Print a small summary table. | rewrite |
| §3 Features | Define `build_UZ(df) -> (U, Z)` that applies optional PCA + `JOINT_TARGETS`. Build `UZ` for every loaded episode. PCA, when enabled, is fit on the pooled *frames* of the training episodes only (still a per-frame transform — no temporal stitching). | rewrite |
| §4 LSSM definition | Unchanged. | — |
| §5 Train | Hardcoded 50/50 slice over `sorted(episodes.keys())` → `train_ts`, `test_ts`. Call `model.fit([UZ[t][0] for t in train_ts], [UZ[t][1] for t in train_ts], ...)`. | rewrite |
| §6a Dynamics | Model-only (eigenvalues, singular values). Unchanged. | — |
| §6b Multi-episode aggregate (new) | Per-episode metric table + summary plots across all loaded episodes. | new |
| §6c Active episode (new) | The one-line switch: `ACTIVE_EPISODE = '...'`. Builds `ep`. | new |
| §6d Prediction plots | Reads `ep`. | edit |
| §6e Cross-correlation | Reads `ep`. | edit |
| §6f Joint plot + 3D arm animation | Reads `ep`. | edit |

### §1 Config

```python
EPISODE_TIMESTAMPS = None   # None -> load all training_data_*.csv;
                            # or explicit list, e.g. ['20260425_011156', ...]
```

Drop `TRAINING_DATA_FILE`. Keep all other config (PCA flags, model hyperparams, `VIZ_DOWNSAMPLE_N`, etc.). No `ACTIVE_EPISODE` or `TRAIN_TIMESTAMPS` here — both live in dedicated cells below so they can be edited without rerunning earlier stages.

### §2 Load

- Resolve `EPISODE_TIMESTAMPS`:
  - `None` → glob `training_data/training_data_*.csv`, extract timestamps, sort.
  - list → use as-is; missing files raise `FileNotFoundError`.
- For each timestamp, read CSV into `episodes[ts]`.
- Derive `emg_cols` from the first episode. For every other episode, assert its EMG column set matches; mismatch raises `ValueError` naming the offending timestamp(s).
- Print summary: timestamp, row count, duration (s), EMG channel count.

### §3 Feature setup

```python
def build_UZ(df, pca_emg=None, pca_joint=None):
    ...
    return U, Z
```

- `pca_emg`, `pca_joint`: pre-fit `PCA` instances or `None`.
- If `USE_PCA_EMG`: fit a single `PCA` on `np.vstack([episodes[t][emg_cols].values for t in train_ts])` — but `train_ts` isn't known until §5, so we either (a) fit PCA in §5 after the slice, or (b) fit PCA over *all* loaded episodes' frames in §3.
  - **Decision:** fit PCA in §3 over all loaded episodes' frames. Rationale: PCA fit on all frames is principled when no temporal information is used; deferring to §5 forces a cell-order coupling for a feature that's off by default. If a clean train-only PCA is later required, it can be moved.
- Same logic for `USE_PCA_JOINT`.
- Build `UZ[ts] = build_UZ(episodes[ts], pca_emg, pca_joint)` for every loaded ts.

### §5 Train

```python
timestamps = sorted(episodes.keys())
split = len(timestamps) // 2
train_ts = timestamps[:split]
test_ts  = timestamps[split:]

U_train = [UZ[t][0] for t in train_ts]
Z_train = [UZ[t][1] for t in train_ts]

model = LSSM(state_dim=STATE_DIM, input_dim=U_train[0].shape[1], output_dim=Z_train[0].shape[1])
model.fit(U_train, Z_train, ...)
```

User can edit the split inline if they want a different partition. Edge case: if only one episode is loaded, `train_ts == []` (because `1 // 2 == 0`). We special-case this: if `len(timestamps) == 1`, train on it and put it in `test_ts` as well, with a printed note.

### §6b Multi-episode aggregate (new)

For each `ts in timestamps`:

- `U, Z = UZ[ts]`; `Z_hat = model.predict(U)`. `LSSM.predict` runs `_forward` from `x0_learned` without touching the streaming `self.x`, so successive `predict` calls across episodes are independent — no manual reset needed.
- Compute per-target MSE, R², Pearson r.

Output:

1. **Table** (printed via `pd.DataFrame`): rows = episodes, columns = `train` flag + per-target metrics.
2. **Bar chart**: per-episode MSE per joint target, colored by train vs. test.
3. **Pooled scatter**: predicted vs. true per target, colored by episode. One subplot per target.

This cell is the "multi-episode test results" view. It does not produce a 3D arm animation — that requires per-frame trajectories and only makes sense per-episode.

### §6c Active episode (new)

```python
ACTIVE_EPISODE = train_ts[0]   # edit this one line

from types import SimpleNamespace
def get_episode(ts):
    df = episodes[ts]
    U, Z = UZ[ts]
    Z_hat = model.predict(U)
    return SimpleNamespace(ts=ts, df=df, U=U, Z=Z, Z_hat=Z_hat,
                           full_joint_data=df[joint_cols_all].values)

ep = get_episode(ACTIVE_EPISODE)
print(f'Active episode: {ep.ts}  (train={ep.ts in train_ts})')
```

To switch episodes for all single-ep viz cells: change the `ACTIVE_EPISODE` string in this cell and rerun §6c through §6f. No model refit.

### §6d / §6e / §6f

Mechanical edits: replace references to `U`, `Z`, `Z_hat`, `full_joint_data`, `df_model['time']` with `ep.U`, `ep.Z`, `ep.Z_hat`, `ep.full_joint_data`, `ep.df['time']`. Plot titles include `ep.ts` so saved figures are identifiable.

## Open implementation details (decided)

- **EMG channel mismatch** → hard error at §2.
- **PCA fit scope** → all loaded episodes' frames pooled (rationale above).
- **Single-episode edge case** → train on it, mark it as both train and test, print a note.
- **Active episode default** → `train_ts[0]` (first sorted training timestamp).
- **`model.predict` between episodes** → uses `x0_learned` afresh each call; no state leaks across episodes in `predict`.

## Testing

This is notebook work, not library work — the verification path is "Run All" + visually inspect:

1. With `EPISODE_TIMESTAMPS = None`, Run All. Confirm:
   - §2 lists all CSVs in `training_data/` with consistent EMG channels.
   - §5 prints a train/test split with roughly half of episodes in each.
   - §6b table renders with train/test flags and finite metrics.
   - §6c through §6f produce plots using the default `ACTIVE_EPISODE`.
2. Edit `ACTIVE_EPISODE` to a test-set timestamp, rerun §6c–§6f, confirm plots refresh without retraining.
3. Set `EPISODE_TIMESTAMPS = ['<one_ts>']`, Run All, confirm the single-episode special case prints its note and §6b/§6f still render.
4. Manually corrupt the EMG channel set of one CSV (or simulate by editing the loader to drop a column for one ts), confirm §2 raises with a clear message. (Optional — revert after.)
