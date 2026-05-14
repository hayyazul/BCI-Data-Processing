import argparse
import cv2
import numpy as np
from collections import deque
import os
import time
import warnings
from typing import List

from models import (Position, get_camera_intrinsics,
                    tag_facing_forward, tag_facing_right, tag_facing_left,
                    tag_facing_back, tag_facing_up, tag_facing_down)
from estimator import MonocularPoseEstimator
from visualization import SimplePlotUpdater, show_playback
from playback_mode import run_playback, run_accept

warnings.filterwarnings("ignore", message=".*more than one new minima.*")


# ==============================================================================
# MAIN TRACKING LOOP
# ==============================================================================

def run_tracking(positions: List[Position], camera_id=0, hfov_deg=70,
                 motion_optimized=True, buffer_seconds=10):
    
    cap = cv2.VideoCapture(camera_id)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    #cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    #cap.set(cv2.CAP_PROP_GAIN, 100)
    
    print(f"Camera: {width}x{height} | Fast shutter")
    
    intrinsics = get_camera_intrinsics(width, height, hfov_deg)
    estimator = MonocularPoseEstimator(intrinsics, motion_optimized=motion_optimized)
    
    all_tag_configs = []
    for pos in positions:
        all_tag_configs.extend(pos.tags)
    
    print(f"\nTracking: {[p.name for p in positions]}")
    for pos in positions:
        for tag in pos.tags:
            off = tag.mount_offset.flatten()
            print(f"  Tag {tag.tag_id}: {tag.tag_family}, {tag.tag_size*1000:.0f}mm, "
                  f"offset=({off[0]:.3f},{off[1]:.3f},{off[2]:.3f})m")
    
    # Create plot updater
    plotter = SimplePlotUpdater([p.name for p in positions],
                                buffer_seconds=buffer_seconds,
                                hfov_deg=hfov_deg)
    
    # State
    recording = False
    recorded = []
    record_start = None
    mirror = True

    fps_buf = deque(maxlen=30)
    last_t = time.time()

    print("\nSPACE = record | M = mirror toggle | Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        fps_buf.append(1.0 / max(now - last_t, 0.001))
        last_t = now
        fps = np.mean(fps_buf)

        if mirror:
            frame = cv2.flip(frame, 1)
        
        # Detect
        tags = estimator.detect_tags_with_size(frame, all_tag_configs)
        all_ests = []
        for pos in positions:
            all_ests.extend(estimator.estimate_position(pos, tags, now))
        
        # Update plot data (cheap - just appends to deque)
        for est in all_ests:
            plotter.add_point(est.position_name, 
                            est.translation[0,0], est.translation[1,0], est.translation[2,0],
                            est.timestamp)
        
        # Update plot display (only redraws every N frames)
        plotter.update(now)
        
        # Record
        if recording:
            recorded.extend(all_ests)
        
        # Draw camera view
        for tag in tags:
            corners = np.int32(tag.corners)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            c = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(frame, c, 4, (0, 0, 255), -1)
            if tag.pose_t is not None:
                z = float(np.asarray(tag.pose_t).flatten()[2])
                cv2.putText(frame, f"T{tag.tag_id} z:{z:.2f}", (c[0]-40, c[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        # HUD
        y = 25
        for pos in positions:
            ests = [e for e in all_ests if e.position_name == pos.name]
            if ests:
                best = max(ests, key=lambda e: e.confidence)
                t = best.translation.flatten()
                cv2.putText(frame, f"{pos.name}: ({t[0]:.2f},{t[1]:.2f},{t[2]:.2f}) [{len(ests)}t]",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                y += 20
        
        status = "REC" if recording else "LIVE"
        sc = (0,0,255) if recording else (0,255,0)
        mir_txt = "MIR" if mirror else "NOMIR"
        cv2.putText(frame, f"{status} {mir_txt} FPS:{fps:.0f} Tags:{[t.tag_id for t in tags]}",
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)
        cv2.putText(frame, "SPACE:rec M:mirror Q:quit", (width-240, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
        
        cv2.imshow("Tracker", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirror = not mirror
            print(f"Mirror: {'ON' if mirror else 'OFF'}")
        elif key == ord(' '):
            if not recording:
                recording = True
                recorded = []
                record_start = now
                print(f">>> RECORDING... SPACE to stop")
            else:
                recording = False
                dur = now - record_start
                print(f">>> Stopped ({dur:.1f}s, {len(recorded)} pts)")
                show_playback(recorded, record_start)
                print("Playback closed.")
    
    plotter.close()
    cap.release()
    cv2.destroyAllWindows()


# ==============================================================================
# RECORD MODE
# ==============================================================================

def run_record(camera_id=0, output_dir="."):
    """Capture raw video for offline AprilTag detection.

    Writes MJPEG-in-AVI: every frame is intra-coded, so detection quality is
    uniform across the file and there are no inter-frame compression artifacts
    that confuse the tag decoder. Sidecar CSV records the Unix timestamp of
    each frame's capture, since the file's nominal FPS rarely matches reality.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}")

    # Ask the camera to deliver MJPG natively — most webcams expose higher
    # FPS in MJPG than in raw YUV, and we're going to re-encode as MJPG anyway.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nominal_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start_unix = time.time()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"recording_{int(start_unix)}")
    video_path = base + ".avi"
    times_path = base + "_timestamps.csv"

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(video_path, fourcc, nominal_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")

    times_file = open(times_path, "w", buffering=1)  # line-buffered: survives crashes
    times_file.write("frame,unix_timestamp\n")

    print(f"Recording {width}x{height} @ {nominal_fps:.0f} fps (nominal) -> {video_path}")
    print(f"Timestamps -> {times_path}")
    print(f"Start (Unix): {start_unix:.6f}")
    print("Q = stop\n")

    fps_buf = deque(maxlen=30)
    last_t = start_unix
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            now = time.time()
            writer.write(frame)
            times_file.write(f"{frame_idx},{now:.6f}\n")
            frame_idx += 1

            fps_buf.append(1.0 / max(now - last_t, 1e-6))
            last_t = now

            # Minimal overlay on a copy so the recorded file stays clean.
            preview = frame.copy()
            elapsed = now - start_unix
            cv2.putText(preview,
                        f"REC {elapsed:6.1f}s  FPS:{np.mean(fps_buf):4.1f}  frames:{frame_idx}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(preview, "Q to stop",
                        (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Recording (raw frames written to disk)", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        end_t = time.time()
        writer.release()
        times_file.close()
        cap.release()
        cv2.destroyAllWindows()
        dur = end_t - start_unix
        avg_fps = frame_idx / dur if dur > 0 else 0.0
        print(f"\nStopped. {frame_idx} frames in {dur:.2f}s ({avg_fps:.1f} fps avg)")
        print(f"Video:      {video_path}")
        print(f"Timestamps: {times_path}")


# ==============================================================================
# CONFIG / ENTRY POINT
# ==============================================================================

def _build_default_positions(tag_size=0.0762):
    inch = 0.0254
    half_thickness_y = 1.5 * inch
    half_thickness_x = 2.0 * inch

    bracelet = Position("bracelet", [
        tag_facing_up   (tag_id=7,  offset=[0, 0, half_thickness_x], tag_size=tag_size),
        # tag_facing_left (tag_id=2,  offset=[-half_thickness_x, 0, 0], tag_size=tag_size),
        tag_facing_forward (tag_id=10,  offset=[0, 0, half_thickness_y], tag_size=tag_size)
    ])
    elbow = Position("elbow", [
        tag_facing_up   (tag_id=1,  offset=[0,  half_thickness_y, 0], tag_size=tag_size),
        tag_facing_down (tag_id=0,  offset=[0, -half_thickness_y, 0], tag_size=tag_size),
        tag_facing_left (tag_id=6,  offset=[-half_thickness_x, 0, 0], tag_size=tag_size),
    ])
    shoulder = Position("shoulder", [
        tag_facing_forward(tag_id=8, offset=[0, 0, 0], tag_size=tag_size)
        # tag_facing_forward(tag_id=9, offset=[0, 0, 0], tag_size=tag_size),
    ])
    return [bracelet, elbow, shoulder]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AprilTag tracker / recorder")
    parser.add_argument("--mode", choices=["visualize", "record", "playback", "accept"],
                        default="visualize",
                        help="visualize: live camera + 3D plot. "
                             "record: raw video only. "
                             "playback: replay a video with a live tuner UI. "
                             "accept: batch-process video with saved settings -> labeled CSV.")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--hfov", type=float, default=70, help="camera horizontal FOV in degrees")
    parser.add_argument("--output", type=str, default=None,
                        help="record: output directory. "
                             "accept: output CSV path or directory (defaults next to video).")
    parser.add_argument("--video", type=str, default=None,
                        help="path to a recorded video (required for playback / accept modes)")
    parser.add_argument("--config", type=str, default="playback_config.json",
                        help="JSON config file for playback-mode tuner settings "
                             "(also consumed by accept mode)")
    args = parser.parse_args()

    if args.mode == "record":
        run_record(camera_id=args.camera, output_dir=args.output or ".")
    elif args.mode == "playback":
        if not args.video:
            parser.error("--video is required for playback mode")
        positions = _build_default_positions()
        run_playback(args.video, positions, config_path=args.config,
                     hfov_deg=args.hfov, buffer_seconds=10)
    elif args.mode == "accept":
        if not args.video:
            parser.error("--video is required for accept mode")
        positions = _build_default_positions()
        run_accept(args.video, positions, config_path=args.config,
                   hfov_deg=args.hfov, output=args.output)
    else:
        positions = _build_default_positions()
        run_tracking(positions, camera_id=args.camera, hfov_deg=args.hfov,
                     motion_optimized=True, buffer_seconds=10)