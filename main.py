import cv2
import numpy as np
from collections import deque
import time
import warnings
from typing import List

from models import (Position, get_camera_intrinsics,
                    tag_facing_forward, tag_facing_right, tag_facing_left,
                    tag_facing_back, tag_facing_up, tag_facing_down)
from estimator import MonocularPoseEstimator
from visualization import SimplePlotUpdater, show_playback

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
    plotter = SimplePlotUpdater([p.name for p in positions], buffer_seconds=buffer_seconds)
    
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
# CONFIG / ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    CAMERA_ID = 0
    CAMERA_HFOV = 70
    TAG_SIZE = 0.0762  # 3 inches – adjust to your actual printed tag size

    # Bracelet geometry (all dimensions in meters)
    inch = 0.0254
    half_thickness_y = 1.5 * inch   # 3-inch sides are 1.5 in from centre
    half_thickness_x = 2.0 * inch   # 2-inch sides are 2.0 in from centre

    bracelet = Position("bracelet", [
        # ID 7 – long side, +Y direction (up)
        tag_facing_up   (tag_id=7,  offset=[0,  half_thickness_y, 0], tag_size=TAG_SIZE),
        # ID 10 – short side, +X direction (right)
        tag_facing_right(tag_id=10, offset=[ half_thickness_x, 0, 0], tag_size=TAG_SIZE),
        # ID 9 – long side, -Y direction (down)
        tag_facing_down (tag_id=9,  offset=[0, -half_thickness_y, 0], tag_size=TAG_SIZE),
        # ID 2 – short side, -X direction (left)
        tag_facing_left (tag_id=2,  offset=[-half_thickness_x, 0, 0], tag_size=TAG_SIZE),
    ])

    shoulder = Position("shoulder", [
        tag_facing_forward (tag_id=1,  offset=[0, 0, 0], tag_size=TAG_SIZE),
    ])

    run_tracking([bracelet, shoulder], camera_id=CAMERA_ID, hfov_deg=CAMERA_HFOV,
                 motion_optimized=True, buffer_seconds=10)