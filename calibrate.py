import cv2
from pupil_apriltags import Detector
import numpy as np

detector = Detector(
    families="tag36h11",  # Changed to match your phone's tag
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

# Open your camera
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("AprilTag Detection - tag36h11")
print("Press 'q' to quit, 'm' to toggle mirror, 'p' to toggle phone screen optimization")

# Phone screen optimization mode (helps with glare/brightness)
phone_mode = True
mirror_mode = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Mirror the frame
    if mirror_mode:
        frame = cv2.flip(frame, 1)

    # If phone mode is on, adjust contrast to better detect screen-displayed tags
    if phone_mode:
        # Increase contrast and brightness slightly for phone screens
        frame_processed = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        gray = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    tags = detector.detect(
        gray,
        estimate_tag_pose=False,
        camera_params=None,
        tag_size=None
    )

    # Draw detected tags
    for tag in tags:
        corners = np.int32(tag.corners)

        # Draw tag outline with thicker line for better visibility
        cv2.polylines(frame, [corners], True, (0, 255, 0), 3)

        # Draw corners
        for i, corner in enumerate(corners):
            cv2.circle(frame, tuple(corner), 6, (255, 0, 0), -1)
            # Optional: Label corners
            cv2.putText(frame, str(i), (corner[0] + 5, corner[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw center
        center = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(frame, center, 8, (0, 0, 255), -1)

        # Tag information display
        cv2.putText(frame, f"ID: {tag.tag_id} (36h11)",
                    (center[0] - 50, center[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Decision margin (detection quality indicator)
        cv2.putText(frame, f"Quality: {tag.decision_margin:.1f}",
                    (center[0] - 50, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Size info
        width = np.linalg.norm(corners[0] - corners[1])
        cv2.putText(frame, f"Size: {width:.0f}px",
                    (center[0] - 50, center[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Display information
    cv2.putText(frame, f"Tags Found: {len(tags)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Show current modes
    mode_text = f"Phone Mode: {'ON' if phone_mode else 'OFF'} | Mirror: {'ON' if mirror_mode else 'OFF'}"
    cv2.putText(frame, mode_text,
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Tips for phone detection
    if len(tags) == 0:
        cv2.putText(frame, "Tips: Reduce phone brightness, avoid glare, hold steady",
                    (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # Show the frame
    cv2.imshow("AprilTag Detection - tag36h11", frame)

    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mirror_mode = not mirror_mode
        print(f"Mirror mode: {'ON' if mirror_mode else 'OFF'}")
    elif key == ord('p'):
        phone_mode = not phone_mode
        print(f"Phone mode: {'ON' if phone_mode else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")