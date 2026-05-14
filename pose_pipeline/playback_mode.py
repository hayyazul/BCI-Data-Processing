"""Playback mode: replay a recorded video through the detector with a live
Tkinter tuner panel. Settings persist to a JSON config so a satisfactory
arrangement is restored on the next run.

Three windows together form the GUI:
  - Tk panel  : detector / preprocessing / estimation sliders + status
  - OpenCV    : video preview with detection overlays + center-position HUD
  - matplotlib: 3D position plot (reuses SimplePlotUpdater)
"""

import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from models import Position, get_camera_intrinsics
from estimator import MonocularPoseEstimator, DetectorParams
from visualization import SimplePlotUpdater


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class PlaybackSettings:
    detector: DetectorParams = field(default_factory=DetectorParams.motion_optimized)
    # Preprocessing
    brightness: int = 0           # added to pixel values (-100 .. 100)
    contrast: float = 1.0         # multiplier (0.3 .. 2.5)
    gamma: float = 1.0            # gamma correction (0.3 .. 3.0)
    clahe: bool = False
    # Estimation
    single_tag_mode: bool = False
    # Playback
    speed: float = 1.0
    loop: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["detector"] = self.detector.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "PlaybackSettings":
        defaults = PlaybackSettings()
        return PlaybackSettings(
            detector=DetectorParams.from_dict(d.get("detector", {})),
            brightness=int(d.get("brightness", defaults.brightness)),
            contrast=float(d.get("contrast", defaults.contrast)),
            gamma=float(d.get("gamma", defaults.gamma)),
            clahe=bool(d.get("clahe", defaults.clahe)),
            single_tag_mode=bool(d.get("single_tag_mode", defaults.single_tag_mode)),
            speed=float(d.get("speed", defaults.speed)),
            loop=bool(d.get("loop", defaults.loop)),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> Optional["PlaybackSettings"]:
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return PlaybackSettings.from_dict(json.load(f))
        except Exception as e:
            print(f"Failed to load config {path}: {e}. Using defaults.")
            return None


# ==============================================================================
# CONTROL PANEL
# ==============================================================================

class ControlPanel:
    """Tk-based tuner. Doesn't run mainloop — the video loop calls .pump()
    each frame to flush Tk events, and reads .current_settings() to apply
    whatever the sliders currently say."""

    def __init__(self, settings: PlaybackSettings, total_frames: int,
                 config_path: str, on_save, on_accept=None):
        self.settings = settings
        self.config_path = config_path
        self.on_save = on_save
        self.on_accept = on_accept
        self.total_frames = max(1, total_frames)

        self.root = tk.Tk()
        self.root.title("AprilTag tuner")
        self.root.geometry("440x820")
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Status strings the video loop pokes each frame.
        self.status_var = tk.StringVar(value="–")
        self.tags_var = tk.StringVar(value="–")
        self.frame_var = tk.StringVar(value=f"0 / {self.total_frames - 1}")

        # Playback
        self.paused_var = tk.BooleanVar(value=False)
        self.loop_var = tk.BooleanVar(value=settings.loop)
        self.speed_var = tk.DoubleVar(value=settings.speed)
        self.seek_var = tk.IntVar(value=0)
        self._programmatic_seek = False
        self._user_dragging_seek = False
        self.seek_request: Optional[int] = None

        # Detector
        d = settings.detector
        self.qd_var = tk.DoubleVar(value=d.quad_decimate)
        self.qs_var = tk.DoubleVar(value=d.quad_sigma)
        self.re_var = tk.BooleanVar(value=d.refine_edges)
        self.ds_var = tk.DoubleVar(value=d.decode_sharpening)
        self.nt_var = tk.IntVar(value=d.nthreads)
        self.dm_var = tk.DoubleVar(value=d.decision_margin_threshold)

        # Preprocessing
        self.br_var = tk.IntVar(value=settings.brightness)
        self.co_var = tk.DoubleVar(value=settings.contrast)
        self.ga_var = tk.DoubleVar(value=settings.gamma)
        self.cl_var = tk.BooleanVar(value=settings.clahe)

        # Estimation
        self.st_var = tk.BooleanVar(value=settings.single_tag_mode)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 2}

        def section(title):
            f = ttk.LabelFrame(self.root, text=title)
            f.pack(fill="x", padx=8, pady=(8, 0))
            return f

        def slider(parent, label, var, lo, hi, step, fmt="{:.2f}"):
            row = ttk.Frame(parent); row.pack(fill="x", **pad)
            ttk.Label(row, text=label, width=18).pack(side="left")
            val_lbl = ttk.Label(row, text=fmt.format(var.get()), width=6)
            val_lbl.pack(side="right")
            tk.Scale(parent, from_=lo, to=hi, resolution=step,
                     orient="horizontal", variable=var,
                     showvalue=False).pack(fill="x", padx=8)
            var.trace_add("write",
                          lambda *_: val_lbl.config(text=fmt.format(var.get())))

        # Playback
        f = section("Playback")
        ttk.Checkbutton(f, text="Pause (SPACE in video window)",
                        variable=self.paused_var).pack(anchor="w", **pad)
        ttk.Checkbutton(f, text="Loop video",
                        variable=self.loop_var).pack(anchor="w", **pad)
        slider(f, "speed", self.speed_var, 0.25, 4.0, 0.05)
        row = ttk.Frame(f); row.pack(fill="x", **pad)
        ttk.Label(row, text="timeline (drag to seek)", width=22).pack(side="left")
        ttk.Label(row, textvariable=self.frame_var).pack(side="right")
        seek_scale = tk.Scale(f, from_=0, to=self.total_frames - 1,
                              orient="horizontal", variable=self.seek_var,
                              showvalue=False, command=self._on_seek)
        seek_scale.pack(fill="x", padx=8)
        # Track whether the user is actively dragging so the playback loop
        # doesn't write back to the variable and fight the drag.
        seek_scale.bind("<ButtonPress-1>",   lambda _e: self._begin_drag())
        seek_scale.bind("<ButtonRelease-1>", lambda _e: self._end_drag())

        # Detector
        f = section("Detector")
        slider(f, "quad_decimate",   self.qd_var, 0.5, 4.0, 0.1)
        slider(f, "quad_sigma",      self.qs_var, 0.0, 2.5, 0.05)
        slider(f, "decode_sharpen",  self.ds_var, 0.0, 1.0, 0.05)
        slider(f, "nthreads",        self.nt_var, 1, 8, 1, fmt="{:.0f}")
        slider(f, "margin threshold",self.dm_var, 0.0, 100.0, 1.0, fmt="{:.0f}")
        ttk.Checkbutton(f, text="refine_edges",
                        variable=self.re_var).pack(anchor="w", **pad)

        # Preprocessing
        f = section("Preprocessing")
        slider(f, "brightness", self.br_var, -100, 100, 1, fmt="{:.0f}")
        slider(f, "contrast",   self.co_var, 0.3, 2.5, 0.05)
        slider(f, "gamma",      self.ga_var, 0.3, 3.0, 0.05)
        ttk.Checkbutton(f, text="CLAHE (adaptive contrast)",
                        variable=self.cl_var).pack(anchor="w", **pad)

        # Estimation
        f = section("Estimation")
        ttk.Checkbutton(f,
            text="Single-tag mode (sticky → highest confidence)",
            variable=self.st_var).pack(anchor="w", **pad)

        # Status
        f = section("Status")
        ttk.Label(f, textvariable=self.status_var).pack(anchor="w", **pad)
        ttk.Label(f, textvariable=self.tags_var).pack(anchor="w", **pad)

        # Buttons
        f = ttk.Frame(self.root); f.pack(fill="x", padx=8, pady=10)
        ttk.Button(f, text="Save config",
                   command=self._save).pack(side="left")
        ttk.Button(f, text="Reset to defaults",
                   command=self._reset).pack(side="left", padx=8)
        self._accept_btn = ttk.Button(f, text="Accept → CSV",
                                      command=self._accept)
        self._accept_btn.pack(side="right")
        if self.on_accept is None:
            self._accept_btn.state(["disabled"])

    def _on_seek(self, val):
        if self._programmatic_seek:
            return
        try:
            self.seek_request = int(float(val))
        except (TypeError, ValueError):
            pass

    def _begin_drag(self):
        self._user_dragging_seek = True

    def _end_drag(self):
        self._user_dragging_seek = False
        # Commit the final drop position once on release.
        try:
            self.seek_request = int(self.seek_var.get())
        except (TypeError, ValueError):
            pass

    def is_user_dragging_seek(self) -> bool:
        return self._user_dragging_seek

    def set_displayed_frame(self, frame_idx: int):
        # Don't fight the user mid-drag — let their drag drive the slider.
        if self._user_dragging_seek:
            self.frame_var.set(f"{frame_idx} / {self.total_frames - 1}")
            return
        self._programmatic_seek = True
        try:
            self.seek_var.set(frame_idx)
        finally:
            self._programmatic_seek = False
        self.frame_var.set(f"{frame_idx} / {self.total_frames - 1}")

    def request_seek(self, frame_idx: int) -> None:
        self.seek_request = max(0, min(self.total_frames - 1, int(frame_idx)))

    def consume_seek_request(self) -> Optional[int]:
        r, self.seek_request = self.seek_request, None
        return r

    def current_settings(self) -> PlaybackSettings:
        s = self.settings
        s.detector = DetectorParams(
            quad_decimate=float(self.qd_var.get()),
            quad_sigma=float(self.qs_var.get()),
            refine_edges=bool(self.re_var.get()),
            decode_sharpening=float(self.ds_var.get()),
            nthreads=int(self.nt_var.get()),
            decision_margin_threshold=float(self.dm_var.get()),
        )
        s.brightness = int(self.br_var.get())
        s.contrast = float(self.co_var.get())
        s.gamma = float(self.ga_var.get())
        s.clahe = bool(self.cl_var.get())
        s.single_tag_mode = bool(self.st_var.get())
        s.speed = float(self.speed_var.get())
        s.loop = bool(self.loop_var.get())
        return s

    def is_paused(self) -> bool:
        return bool(self.paused_var.get())

    def toggle_paused(self):
        self.paused_var.set(not self.paused_var.get())

    def update_status(self, *, fps: float, detect_ms: float, n_tags: int,
                      n_pos_with_estimate: int, n_pos_total: int):
        self.status_var.set(
            f"FPS: {fps:5.1f}   detect: {detect_ms:5.1f} ms   "
            f"positions: {n_pos_with_estimate}/{n_pos_total}")
        self.tags_var.set(f"Tags this frame: {n_tags}")

    def pump(self):
        try:
            self.root.update()
        except tk.TclError:
            self._closed = True

    def is_closed(self) -> bool:
        return self._closed

    def _on_close(self):
        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _save(self):
        try:
            self.on_save()
            self.status_var.set(f"Saved → {self.config_path}")
        except Exception as e:
            self.status_var.set(f"Save failed: {e}")

    def _accept(self):
        if self.on_accept is None:
            return
        # Disable the button so a second click can't re-enter while processing.
        self._accept_btn.state(["disabled"])
        try:
            self.status_var.set("Accept: processing video...")
            self.root.update_idletasks()
            result = self.on_accept()
            if result:
                self.status_var.set(f"Accepted → {result}")
            else:
                self.status_var.set("Accept finished.")
        except Exception as e:
            self.status_var.set(f"Accept failed: {e}")
        finally:
            try:
                self._accept_btn.state(["!disabled"])
            except tk.TclError:
                pass

    def _reset(self):
        defaults = PlaybackSettings()
        self.qd_var.set(defaults.detector.quad_decimate)
        self.qs_var.set(defaults.detector.quad_sigma)
        self.re_var.set(defaults.detector.refine_edges)
        self.ds_var.set(defaults.detector.decode_sharpening)
        self.nt_var.set(defaults.detector.nthreads)
        self.dm_var.set(defaults.detector.decision_margin_threshold)
        self.br_var.set(defaults.brightness)
        self.co_var.set(defaults.contrast)
        self.ga_var.set(defaults.gamma)
        self.cl_var.set(defaults.clahe)
        self.st_var.set(defaults.single_tag_mode)
        self.speed_var.set(defaults.speed)
        self.loop_var.set(defaults.loop)


# ==============================================================================
# PREPROCESSING + HUD HELPERS
# ==============================================================================

def _apply_preprocessing(frame: np.ndarray, s: PlaybackSettings) -> np.ndarray:
    out = frame
    if s.brightness != 0 or abs(s.contrast - 1.0) > 1e-3:
        out = cv2.convertScaleAbs(out, alpha=s.contrast, beta=s.brightness)
    if abs(s.gamma - 1.0) > 1e-3:
        inv = 1.0 / max(s.gamma, 1e-3)
        lut = np.array([((i / 255.0) ** inv) * 255
                        for i in range(256)]).astype("uint8")
        out = cv2.LUT(out, lut)
    if s.clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
        out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
    return out


def _project(p3, K: np.ndarray) -> Optional[Tuple[int, int]]:
    z = float(p3[2])
    if z < 1e-3:
        return None
    u = K[0, 0] * float(p3[0]) / z + K[0, 2]
    v = K[1, 1] * float(p3[1]) / z + K[1, 2]
    return (int(round(u)), int(round(v)))


_PALETTE_BGR = [(0, 200, 255), (0, 255, 120), (255, 120, 0),
                (255, 0, 200), (200, 200, 0), (0, 120, 255)]

def _bgr_color_for(idx: int):
    return _PALETTE_BGR[idx % len(_PALETTE_BGR)]


def _draw_center_overlay(frame, position_estimates, position_index,
                         K: np.ndarray, tag_pixel_centers: Dict[int, Tuple[int, int]]):
    """Per Position: project the mean-translation 'center' to pixels and
    draw connecting lines to each contributing AprilTag's pixel center."""
    by_name: Dict[str, list] = {}
    for est in position_estimates:
        by_name.setdefault(est.position_name, []).append(est)

    for name, ests in by_name.items():
        idx = position_index.get(name, 0)
        color = _bgr_color_for(idx)
        ts = np.stack([e.translation.flatten() for e in ests], axis=0)
        center3 = ts.mean(axis=0)
        center_px = _project(center3, K)
        if center_px is None:
            continue
        for e in ests:
            tag_px = tag_pixel_centers.get(e.source_tag_id)
            if tag_px is None:
                continue
            cv2.line(frame, tag_px, center_px, color, 2, cv2.LINE_AA)
        cv2.circle(frame, center_px, 8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, center_px, 6, color, -1, cv2.LINE_AA)
        cv2.putText(frame, name, (center_px[0] + 10, center_px[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ==============================================================================
# MAIN PLAYBACK LOOP
# ==============================================================================

def run_playback(video_path: str, positions: List[Position],
                 config_path: str = "playback_config.json",
                 hfov_deg: float = 70.0,
                 buffer_seconds: float = 10.0):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video: {video_path}  {width}x{height}  {src_fps:.1f} fps  "
          f"{total_frames} frames")

    intrinsics = get_camera_intrinsics(width, height, hfov_deg)
    K = intrinsics.camera_matrix

    loaded = PlaybackSettings.load(config_path)
    settings = loaded or PlaybackSettings()
    print(f"Config: {config_path} ({'loaded' if loaded else 'defaults'})")

    estimator = MonocularPoseEstimator(intrinsics, params=settings.detector)
    all_tag_configs = []
    for pos in positions:
        all_tag_configs.extend(pos.tags)
    position_index = {p.name: i for i, p in enumerate(positions)}

    # Tk root must exist before matplotlib's TkAgg figure to keep one main loop.
    panel: Optional[ControlPanel] = None

    def _save():
        if panel is not None:
            panel.current_settings().save(config_path)
            print(f"Saved {config_path}")

    def _accept_now():
        # Pause playback, persist live settings, then run the batch pass on
        # the current video using the same config_path. Tk events are pumped
        # via the progress callback so the UI stays responsive.
        if panel is None:
            return None
        panel.paused_var.set(True)
        panel.current_settings().save(config_path)

        def _progress(idx, total, n_rows):
            try:
                panel.status_var.set(
                    f"Accept: {idx}/{total} frames, {n_rows} rows")
                panel.root.update()
            except tk.TclError:
                pass

        return run_accept(video_path, positions, config_path=config_path,
                          hfov_deg=hfov_deg, progress_callback=_progress)

    panel = ControlPanel(settings, total_frames, config_path,
                         on_save=_save, on_accept=_accept_now)
    plotter = SimplePlotUpdater([p.name for p in positions],
                                buffer_seconds=buffer_seconds, hfov_deg=hfov_deg)

    frame_idx = 0
    last_frame = None
    last_tags: list = []
    last_estimates: list = []
    last_tag_pixel_centers: Dict[int, Tuple[int, int]] = {}
    fps_buf = deque(maxlen=30)
    last_wall = time.time()
    next_frame_due = time.time()
    detect_ms = 0.0

    print("\nVideo window keys:")
    print("  SPACE  pause / play")
    print("  , / .  step ±1 frame   (auto-pauses)")
    print("  [ / ]  step ±10 frames")
    print("  j / l  step ±1 second")
    print("  0      jump to first frame")
    print("  Q      quit")
    print("Tk panel: drag the timeline to seek; tune detector/preprocessing.\n")

    try:
        while True:
            if panel.is_closed():
                break

            s = panel.current_settings()
            estimator.set_params(s.detector)

            # Decide whether to advance frame.
            seek_to = panel.consume_seek_request()
            advance = False
            if seek_to is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                advance = True
                estimator.reset_sticky_selection()
            elif panel.is_paused():
                advance = (last_frame is None)
            else:
                now = time.time()
                target_dt = (1.0 / max(src_fps, 1.0)) / max(s.speed, 0.05)
                if now >= next_frame_due:
                    advance = True
                    next_frame_due = max(next_frame_due + target_dt, now)

            re_detect = False
            if advance:
                ret, frame = cap.read()
                if not ret:
                    if s.loop and total_frames > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        estimator.reset_sticky_selection()
                        next_frame_due = time.time()
                        panel.pump()
                        continue
                    panel.paused_var.set(True)
                else:
                    last_frame = frame
                    frame_idx = max(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, 0)
                    panel.set_displayed_frame(frame_idx)
                    re_detect = True

            if last_frame is None:
                panel.pump()
                time.sleep(0.01)
                continue

            # When paused, re-run detection if any settings could affect it.
            # Cheaper than tracking dirty bits: always re-detect on each tick
            # while paused. The user explicitly paused so detection cost is fine.
            if panel.is_paused() and not re_detect:
                re_detect = True

            proc = _apply_preprocessing(last_frame, s)

            if re_detect:
                t0 = time.time()
                tags = estimator.detect_tags_with_size(proc, all_tag_configs)
                detect_ms = (time.time() - t0) * 1000.0
                last_tags = tags
                last_tag_pixel_centers = {
                    t.tag_id: (int(t.center[0]), int(t.center[1])) for t in tags
                }
                now = time.time()
                ests = []
                for pos in positions:
                    pe = estimator.estimate_position(pos, tags, now)
                    if s.single_tag_mode:
                        pe = estimator.select_single_estimate(pos.name, pe)
                    ests.extend(pe)
                last_estimates = ests
                if advance:  # don't pollute history while paused & tweaking
                    for est in ests:
                        plotter.add_point(est.position_name,
                                          est.translation[0, 0],
                                          est.translation[1, 0],
                                          est.translation[2, 0],
                                          est.timestamp)

            plotter.update(time.time())

            # ---- HUD ----
            disp = proc.copy()
            for tag in last_tags:
                corners = np.int32(tag.corners)
                cv2.polylines(disp, [corners], True, (0, 255, 0), 2)
                c = (int(tag.center[0]), int(tag.center[1]))
                cv2.circle(disp, c, 4, (0, 0, 255), -1)
                if tag.pose_t is not None:
                    z = float(np.asarray(tag.pose_t).flatten()[2])
                    cv2.putText(disp, f"T{tag.tag_id} z:{z:.2f}",
                                (c[0] - 40, c[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 0), 1)

            _draw_center_overlay(disp, last_estimates, position_index, K,
                                 last_tag_pixel_centers)

            now = time.time()
            fps_buf.append(1.0 / max(now - last_wall, 1e-3))
            last_wall = now
            live_fps = float(np.mean(fps_buf))
            n_with = len({e.position_name for e in last_estimates})
            panel.update_status(fps=live_fps, detect_ms=detect_ms,
                                n_tags=len(last_tags),
                                n_pos_with_estimate=n_with,
                                n_pos_total=len(positions))

            y = 25
            for pos in positions:
                ests = [e for e in last_estimates if e.position_name == pos.name]
                if ests:
                    best = max(ests, key=lambda e: e.confidence)
                    t = best.translation.flatten()
                    cv2.putText(disp,
                        f"{pos.name}: ({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f}) "
                        f"[{len(ests)}t id={best.source_tag_id}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        _bgr_color_for(position_index[pos.name]), 1)
                    y += 20

            mode_str = "PAUSED" if panel.is_paused() else f"{s.speed:.2f}x"
            single = " SINGLE" if s.single_tag_mode else ""
            cv2.putText(disp,
                f"{mode_str}{single}  frame {frame_idx}/{max(total_frames-1,0)}  "
                f"FPS:{live_fps:.0f}  tags:{[t.tag_id for t in last_tags]}",
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0), 1)
            cv2.putText(disp,
                        "SPACE:pause Q:quit  ,/.: 1f  [/]: 10f  j/l: 1s  0:start",
                        (max(width - 470, 10), height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow("Playback", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:  # no key
                pass
            elif key == ord('q'):
                break
            elif key == ord(' '):
                panel.toggle_paused()
            else:
                # Step keys: comma/period = ±1 frame, [/] = ±10, j/l = ±1 sec,
                # 0 = jump to start. Stepping while playing auto-pauses so you
                # land precisely on the chosen frame.
                step = None
                if key == ord(','):
                    step = -1
                elif key == ord('.'):
                    step = 1
                elif key == ord('['):
                    step = -10
                elif key == ord(']'):
                    step = 10
                elif key == ord('j'):
                    step = -int(round(src_fps))
                elif key == ord('l'):
                    step = int(round(src_fps))
                elif key == ord('0'):
                    panel.request_seek(0)
                    if not panel.is_paused():
                        panel.paused_var.set(True)
                if step is not None:
                    panel.request_seek(frame_idx + step)
                    if not panel.is_paused():
                        panel.paused_var.set(True)

            panel.pump()
    finally:
        try:
            panel.current_settings().save(config_path)
            print(f"Auto-saved settings → {config_path}")
        except Exception as e:
            print(f"Auto-save failed: {e}")
        cap.release()
        try:
            plotter.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        if not panel.is_closed():
            try:
                panel.root.destroy()
            except Exception:
                pass


# ==============================================================================
# ACCEPT MODE: batch-process a video to a labeled pose CSV
# ==============================================================================

def _load_timestamp_sidecar(video_path: str) -> Dict[int, float]:
    """If record mode wrote a `<base>_timestamps.csv` next to the video, load
    it as {frame_idx: unix_timestamp} so the output CSV can carry real wall
    times. Empty dict if no sidecar found."""
    base, _ext = os.path.splitext(video_path)
    times_path = base + "_timestamps.csv"
    if not os.path.exists(times_path):
        return {}
    out: Dict[int, float] = {}
    try:
        with open(times_path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    out[int(row[0])] = float(row[1])
                except ValueError:
                    pass
    except Exception as e:
        print(f"WARN: failed to read {times_path}: {e}")
        return {}
    return out


def _resolve_accept_output_path(video_path: str, output: Optional[str]) -> str:
    """Decide where the CSV goes. If `output` is None, write next to the
    video as `<base>_poses.csv`. If it ends in .csv, treat as full path. If
    it points to a directory (or doesn't end in .csv), write the default
    filename inside it."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    default_name = f"{base}_poses.csv"
    if output is None:
        return os.path.join(os.path.dirname(video_path) or ".", default_name)
    if output.lower().endswith(".csv"):
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        return output
    os.makedirs(output, exist_ok=True)
    return os.path.join(output, default_name)


def run_accept(video_path: str, positions: List[Position],
               config_path: str = "playback_config.json",
               hfov_deg: float = 70.0,
               output: Optional[str] = None,
               progress_callback=None) -> str:
    """Process the full video using the saved playback config and write a
    labeled CSV of pose estimates. Returns the output path.

    CSV is in long format — one row per (frame, position) detection. Frames
    with no detected positions produce no rows. If a `<base>_timestamps.csv`
    sidecar exists from record mode, its unix times are joined per frame.

    progress_callback, if given, is called as
        progress_callback(frame_idx, total_frames, n_rows_so_far)
    every few frames — useful for keeping a GUI responsive."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video: {video_path}  {width}x{height}  {src_fps:.1f} fps  "
          f"{total_frames} frames")

    settings = PlaybackSettings.load(config_path)
    if settings is None:
        print(f"WARN: no config at {config_path}; using defaults. "
              f"Tune via --mode playback first for best results.")
        settings = PlaybackSettings()
    else:
        print(f"Config: {config_path} (loaded)")

    intrinsics = get_camera_intrinsics(width, height, hfov_deg)
    estimator = MonocularPoseEstimator(intrinsics, params=settings.detector)
    all_tag_configs = []
    for pos in positions:
        all_tag_configs.extend(pos.tags)

    frame_unix = _load_timestamp_sidecar(video_path)
    has_unix = bool(frame_unix)
    if has_unix:
        print(f"Timestamps sidecar: {len(frame_unix)} entries — including unix_timestamp column.")

    output_path = _resolve_accept_output_path(video_path, output)
    fields = ["frame_idx", "video_time_s"]
    if has_unix:
        fields.append("unix_timestamp")
    fields += ["position_name", "source_tag_id", "confidence", "x", "y", "z"]

    print(f"Output: {output_path}")
    print(f"Settings: detector={settings.detector.to_dict()}  "
          f"single_tag_mode={settings.single_tag_mode}  "
          f"preproc=[brightness={settings.brightness} contrast={settings.contrast} "
          f"gamma={settings.gamma} clahe={settings.clahe}]")

    n_rows = 0
    n_frames_with_detection = 0
    frame_idx = 0
    t_start = time.time()
    next_progress = t_start + 2.0

    try:
        with open(output_path, "w", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(fields)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                proc = _apply_preprocessing(frame, settings)
                tags = estimator.detect_tags_with_size(proc, all_tag_configs)

                video_time_s = frame_idx / src_fps if src_fps > 0 else 0.0
                unix_t = frame_unix.get(frame_idx)

                ests = []
                for pos in positions:
                    pe = estimator.estimate_position(pos, tags, video_time_s)
                    if settings.single_tag_mode:
                        pe = estimator.select_single_estimate(pos.name, pe)
                    ests.extend(pe)

                if ests:
                    n_frames_with_detection += 1

                for est in ests:
                    t = est.translation.flatten()
                    row = [frame_idx, f"{video_time_s:.6f}"]
                    if has_unix:
                        row.append(f"{unix_t:.6f}" if unix_t is not None else "")
                    row += [est.position_name, est.source_tag_id,
                            f"{est.confidence:.3f}",
                            f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}"]
                    writer.writerow(row)
                    n_rows += 1

                frame_idx += 1

                if progress_callback is not None and (frame_idx % 5) == 0:
                    try:
                        progress_callback(frame_idx, total_frames, n_rows)
                    except Exception:
                        pass

                now = time.time()
                if now >= next_progress:
                    elapsed = now - t_start
                    rate = frame_idx / elapsed if elapsed > 0 else 0.0
                    remaining = ((total_frames - frame_idx) / rate
                                 if rate > 0 and total_frames > 0 else 0.0)
                    print(f"  {frame_idx}/{total_frames}  "
                          f"({rate:.1f} fps, ~{remaining:.0f}s remaining, "
                          f"{n_rows} rows)")
                    next_progress = now + 2.0
    finally:
        cap.release()

    elapsed = time.time() - t_start
    rate = frame_idx / elapsed if elapsed > 0 else 0.0
    pct = (100.0 * n_frames_with_detection / frame_idx) if frame_idx else 0.0
    print(f"\nDone: {n_rows} pose rows from {n_frames_with_detection}/{frame_idx} "
          f"frames ({pct:.1f}% had a detection) in {elapsed:.1f}s ({rate:.1f} fps).")
    print(f"Output: {output_path}")
    return output_path
