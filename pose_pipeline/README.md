# Pose extraction pipeline

End-to-end pipeline for turning a recorded video of the subject's right arm
(wearing AprilTag markers at the shoulder, elbow, and bracelet) into a clean
per-frame pose CSV that downstream notebooks consume.

## Layout

| File | Role |
|------|------|
| `main.py` | Entry point. Reads a recording (or live camera), runs the AprilTag detector, and writes raw per-frame pose CSV plus playback artifacts. |
| `models.py` | Shared dataclasses: `Position`, `PositionEstimate`, `CameraIntrinsics`, `TagMount`, `get_camera_intrinsics(...)`. |
| `estimator.py` | `MonocularPoseEstimator` — runs `pupil_apriltags` detection and converts tag poses into shoulder/elbow/bracelet positions. |
| `visualization.py` | `SimplePlotUpdater`, `show_playback` — live plot of detected positions and overlay for the playback window. |
| `playback_mode.py` | Tk-based playback/tuning GUI. Replays a recording through the detector with sliders for detector params; persists settings to `playback_config.json`. |
| `playback_config.json` | Persisted detector settings for `playback_mode`. |
| `calibrate.py` | Standalone — runs camera intrinsic calibration from a folder of AprilTag images. |
| `filter.py` | Standalone — takes a raw pose CSV produced by `main.py` and writes the smoothed `*_poses_smoothed.csv` that `data_preprocess.ipynb` reads. |
| `animate.py` | Standalone — animates a pose CSV for visual inspection. |

## Inputs and outputs

- **Inputs:** a recorded video and any associated calibration in `../recordings/`.
- **Outputs:** raw and smoothed per-frame pose CSVs in `../recordings/`,
  consumed by `../data_preprocess.ipynb`.

## Running

The modules use plain top-level imports (`from models import …`,
`from estimator import …`, etc.), so they expect to be run with this folder
on `sys.path`. Two equivalent ways:

```bash
# Option A — from inside the folder
cd pose_pipeline
python main.py --config playback_config.json <args...>

# Option B — from the repo root, treat as a package
python -m pose_pipeline.main --config pose_pipeline/playback_config.json <args...>
```

Typical sequence for a new recording:

1. `python calibrate.py` (once per camera) — produces intrinsics.
2. `python main.py …` — record + detect, dumps raw pose CSV into `../recordings/`.
3. `python filter.py <raw_pose_csv>` — smooth the raw CSV → `*_poses_smoothed.csv`.
4. (optional) `python playback_mode.py …` — re-tune detector params on a recorded video.
5. (optional) `python animate.py <pose_csv>` — sanity-check the trajectory.

Step 3's smoothed CSV is what `../data_preprocess.ipynb` loads.
