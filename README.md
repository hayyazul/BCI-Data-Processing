# BCI Data Processing

End-to-end pipeline for the UIUC SIG Robotics BCI project: record a subject's
right arm with EMG electrodes + AprilTag markers, extract pose, fuse it with
EMG, and train a model that maps EMG → joint angles.

## Pipeline overview

```
                ┌─────────────────────────┐
recordings/ ──► │  pose_pipeline/         │ ──► recordings/*_poses_smoothed.csv
  - video       │  (capture, detect,      │
  - emg CSVs    │   smooth)               │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │  data_preprocess.ipynb  │ ──► training_data/training_data_*.csv
                │  (IK, EMG conditioning, │
                │   alignment, assembly)  │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │  model_training.ipynb   │ ──► trained LSSM + eval plots
                │  (LSSM fit + eval)      │
                └─────────────────────────┘
```

## Repository layout

| Path | Role |
|------|------|
| `pose_pipeline/` | Capture + AprilTag detection + pose smoothing. See `pose_pipeline/README.md` for per-file details. |
| `recordings/` | Per-session raw inputs and pose CSVs. *Not committed.* |
| `data_preprocess.ipynb` | Raw recordings → `df_model` (EMG envelopes + joint angles on a shared timeline). |
| `training_data/` | Output of `data_preprocess.ipynb`. *Not committed.* |
| `model_training.ipynb` | Trains the Numba-jitted LSSM on `df_model` and runs diagnostics. |
| `arm_inverse_kinematics.py` | 4-DOF arm IK + matching FK + FK/IK consistency diagnostic. |
| `arm_visualizer.py` | 3D Plotly arm animation (`ArmVisualizer`) — overlay ground truth + reconstructed arms with a play/pause slider. |
| `claude_scripts/` | Reusable helpers (notebook cell listing/appending). |
| `obsolete/` | Pre-refactor monolithic notebooks (`bci_data_analysis.ipynb` — superseded by the preprocess + training notebooks; `model_testing.ipynb` — earlier subspace-ID LSSM experiment). Kept for reference only. |

## Setup

```bash
# Python env
uv venv                          # or python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt

# Required directories
mkdir -p recordings training_data
```

`recordings/` must contain, per session (`time_stamp` is shared across all three files):

- `camera0_{time_stamp}.mp4` (or equivalent) — video of the subject's right
  arm wearing AprilTag markers at shoulder, elbow, and bracelet.
- `emg_data_{time_stamp}.csv` — raw EMG channels.
- `emg_timestamps_{time_stamp}.csv` — Unix `pc_time` for each EMG packet.

The pose pipeline then writes `camera0_{time_stamp}_poses_smoothed.csv` next
to the video; this is what `data_preprocess.ipynb` reads.

## End-to-end workflow

### 1. Record + extract pose (`pose_pipeline/`)

```bash
cd pose_pipeline
python calibrate.py          # once per camera; produces intrinsics
python main.py …             # record + detect → raw pose CSV in ../recordings/
python filter.py <raw.csv>   # smooth → *_poses_smoothed.csv
```

Optional: `playback_mode.py` for tuning detector params on a recorded clip,
`animate.py` for a quick visual check of a pose CSV. Full details in
`pose_pipeline/README.md`.

### 2. Preprocess into model-ready data (`data_preprocess.ipynb`)

Set `time_stamp` in the config cell to the session you want, then run all.
The notebook is organized as collapsible stages (use JupyterLab's
heading-collapse to focus on one stage at a time):

1. Load raw pose + EMG + EMG packet timestamps
2. Pose → wide per-frame format → 4-DOF joint angles via inverse kinematics
3. EMG ingest: uniform timeline + active-channel detection
4. Pose/EMG overlap window (shared by all downstream stages)
5. EMG conditioning: normalize → rectify → lowpass → z-score
6. Pose alignment: trim and upsample joints to the EMG rate
7. Assemble `df_model = [time, <EMG channels>, q1..q4]`
   - **7b. Arm reconstruction sanity check** — animated 3D arm overlay
     (ground-truth pose vs FK from IK joints). Useful for catching
     IK/FK convention drift before training.
8. Save → `training_data/training_data_{time_stamp}.csv`

All knobs (arm geometry, EMG filter cutoffs, active-channel threshold,
EMG/pose offset, visualization downsample, etc.) are at the top of the
config cell. Intermediate plots and previews are gated behind
`DISPLAY_INTERMEDIATE`.

### 3. Train + evaluate (`model_training.ipynb`)

By default the notebook auto-loads the most recent
`training_data/training_data_*.csv`; set `TRAINING_DATA_FILE` to pin a
specific session. Stages:

1. Config + imports
2. Load preprocessed training data
3. Feature setup — optional PCA on EMG and/or joint side (off by default).
   `JOINT_TARGETS` controls which joints to predict in the no-PCA case
   (default `['q4']`).
4. LSSM definition — Numba-jitted forward/backward passes, analytic
   gradient, L-BFGS-B with multi-restart + polish.
5. Train
6. Evaluation
   - Learned dynamics: eigenvalues, singular values, hidden-state activity
   - True vs predicted target plots
   - Input/output cross-correlation (peak lag per channel pair)
   - Inverse PCA → full joint angles, then 3D FK arm animation
     (ground truth vs EMG-reconstructed) via `ArmVisualizer`

## Conventions

- **Arm chain (4 DOF):** `q1` (shoulder azimuth), `q2` (shoulder elevation),
  `q3` (forearm twist), `q4` (elbow flex). `q4 = 0` is fully bent
  (forearm folded back along upper arm); `q4 = π` is fully extended.
- **Link lengths:** measured per subject; defaults in
  `arm_inverse_kinematics.py` (`L1_FIXED = 14 in`, `L2_FIXED = 10 in`) and
  overridable via `UPPER_ARM_LENGTH_M` / `FOREARM_LENGTH_M` in
  `data_preprocess.ipynb`.
- **IK/FK match:** `forward_kinematics_fixed` is the exact inverse of
  `compute_joint_angles_from_data`. `diagnose_fk_ik(pose_df, joint_angles)`
  runs a battery of consistency checks (link-length mismatch, forearm
  direction angle error, sign-flip test, alternative-convention test) — run
  this whenever you change the IK convention.
- **Timebase:** the preprocessing step expresses both streams in
  seconds-relative-to-`session_epoch`, finds the overlap window, and applies
  the same window everywhere downstream. The final `df_model` is sampled at
  the EMG rate.

## Where to look when something breaks

- **Joint angles look jagged or wrap weirdly** → inspect §2b in
  `data_preprocess.ipynb` and run `diagnose_fk_ik` from
  `arm_inverse_kinematics.py`. The §7b sanity-check animation is the
  fastest visual check.
- **EMG/pose look out of phase** → tune `EMG_POSE_OFFSET_SAMPLES` in
  `data_preprocess.ipynb` and re-check the overlap window in §4.
- **Model converges to a flat prediction** → check eigenvalue magnitudes
  near 1 in evaluation §6a; raise `L2_ALPHA` or change `STATE_DIM`.
- **Detector misses tags** → use `pose_pipeline/playback_mode.py` to
  re-tune detector parameters on the recording, then re-run `main.py`.
