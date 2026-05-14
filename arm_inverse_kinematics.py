"""Kinematics for the 4-DOF shoulder/elbow arm chain.

Provides:
  - compute_joint_angles_from_data: inverse kinematics. Per-frame shoulder,
    elbow, and bracelet positions -> joint angles q1..q4. The ± ambiguity at
    each joint is resolved by picking the candidate closest (circularly) to
    the previous frame's solution.
  - forward_kinematics_fixed: forward kinematics matching the IK convention.
    Joint angles + link lengths -> elbow/wrist positions in shoulder frame.
  - diagnose_fk_ik: battery of consistency checks comparing reconstructed
    elbow/wrist to ground-truth pose; isolates link-length vs forearm-
    direction sources of error.
"""

import numpy as np
import pandas as pd

L1_FIXED = 14 * 0.0254    # 0.3556 m (measured upper arm)
L2_FIXED = 10 * 0.0254    # 0.254 m (measured forearm)


def resolve_sign(candidates_func, prev_val):
    """
    candidates_func returns two possible angle values (in radians).
    Pick the one closest (circularly) to prev_val.
    """
    val1, val2 = candidates_func()
    # Wrap to [-pi, pi)
    val1 = np.arctan2(np.sin(val1), np.cos(val1))
    val2 = np.arctan2(np.sin(val2), np.cos(val2))

    # Circular difference to prev_val
    diff1 = np.abs(np.arctan2(np.sin(val1 - prev_val), np.cos(val1 - prev_val)))
    diff2 = np.abs(np.arctan2(np.sin(val2 - prev_val), np.cos(val2 - prev_val)))

    return val1 if diff1 <= diff2 else val2


def compute_joint_angles_from_data(pose_df, L1=L1_FIXED):
    x1 = pose_df['elbow_x'] - pose_df['shoulder_x']
    y1 = pose_df['elbow_y'] - pose_df['shoulder_y']
    z1 = pose_df['elbow_z'] - pose_df['shoulder_z']

    x2 = pose_df['bracelet_x'] - pose_df['shoulder_x']
    y2 = pose_df['bracelet_y'] - pose_df['shoulder_y']
    z2 = pose_df['bracelet_z'] - pose_df['shoulder_z']

    n = len(pose_df)
    q1 = np.zeros(n); q2 = np.zeros(n); q3 = np.zeros(n); q4 = np.zeros(n)

    prev_q1, prev_q2, prev_q3, prev_q4 = 0.0, np.pi/2, 0.0, np.pi/2

    for i in range(n):
        base_q1 = np.arctan2(y1.iloc[i], x1.iloc[i])
        q1[i] = resolve_sign(lambda: (base_q1, np.arctan2(-y1.iloc[i], x1.iloc[i])), prev_q1)

        r_xy = np.sqrt(x1.iloc[i]**2 + y1.iloc[i]**2)
        q2[i] = resolve_sign(lambda: (np.arctan2(r_xy, z1.iloc[i]), np.arctan2(-r_xy, z1.iloc[i])), prev_q2)

        c1, s1 = np.cos(q1[i]), np.sin(q1[i])
        c2, s2 = np.cos(q2[i]), np.sin(q2[i])
        B1 = x2.iloc[i]*c1*c2 + y2.iloc[i]*s1*c2 - z2.iloc[i]*s2
        B2 = -x2.iloc[i]*c1*s2 - y2.iloc[i]*s1*s2 - z2.iloc[i]*c2
        B3 = -x2.iloc[i]*s1 + y2.iloc[i]*c1

        q3[i] = resolve_sign(lambda: (np.arctan2(B3, B1), np.arctan2(-B3, B1)), prev_q3)

        # q4: forearm projection onto -u is (B2 + L1), not (B2 - L1)
        r_B = np.sqrt(B1**2 + B3**2)
        q4[i] = resolve_sign(lambda: (np.arctan2(r_B, B2 + L1), np.arctan2(-r_B, B2 + L1)), prev_q4)

        prev_q1, prev_q2, prev_q3, prev_q4 = q1[i], q2[i], q3[i], q4[i]

    return pd.DataFrame({
        'video_time_s': pose_df['video_time_s'],
        'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4,
    })


def forward_kinematics_fixed(q1, q2, q3, q4, L1=L1_FIXED, L2=L2_FIXED):
    """
    Matches the IK convention in compute_joint_angles_from_data:
      - Upper-arm frame {e1, e2, -u} where u = shoulder->elbow direction
      - Forearm direction = -cos(q4)*u + sin(q4)*(cos(q3)*e1 + sin(q3)*e2)
      - q4 = 0 means forearm folded back along upper arm (elbow fully bent)
      - q4 = pi means forearm extended straight out from upper arm
    Returns (elbow, wrist) in the shoulder frame.
    """
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c3, s3 = np.cos(q3), np.sin(q3)
    c4, s4 = np.cos(q4), np.sin(q4)

    u = np.array([s2*c1, s2*s1, c2])
    elbow = L1 * u

    e1 = np.array([c1*c2, s1*c2, -s2])
    e2 = np.array([-s1, c1, 0.0])

    forearm_dir = -c4*u + s4*(c3*e1 + s3*e2)
    wrist = elbow + L2 * forearm_dir
    return elbow, wrist


def diagnose_fk_ik(pose_df, joint_angles, L1_fixed=L1_FIXED, L2_fixed=L2_FIXED):
    """
    Run a battery of tests to localize where the IK/FK mismatch is.

    Compares reconstructed elbow/wrist (from joint_angles via FK) to ground
    truth (pose_df). Reports link-length sanity, forearm direction angle
    error, sign-flip test, alternative convention test, and IK-internal
    consistency (B1/B2/B3 vs forearm projections).
    """
    n = len(joint_angles)

    elb_true = np.column_stack([
        pose_df['elbow_x'] - pose_df['shoulder_x'],
        pose_df['elbow_y'] - pose_df['shoulder_y'],
        pose_df['elbow_z'] - pose_df['shoulder_z'],
    ])
    wrs_true = np.column_stack([
        pose_df['bracelet_x'] - pose_df['shoulder_x'],
        pose_df['bracelet_y'] - pose_df['shoulder_y'],
        pose_df['bracelet_z'] - pose_df['shoulder_z'],
    ])

    L1_meas = np.linalg.norm(elb_true, axis=1)
    forearm_true = wrs_true - elb_true
    L2_meas = np.linalg.norm(forearm_true, axis=1)

    print("=== LINK LENGTH SANITY CHECK ===")
    print(f"L1 measured: mean={L1_meas.mean():.4f} m, std={L1_meas.std():.4f}, "
          f"min={L1_meas.min():.4f}, max={L1_meas.max():.4f}")
    print(f"L1 fixed:    {L1_fixed:.4f} m  (= {L1_fixed/0.0254:.2f} inches)")
    print(f"L2 measured: mean={L2_meas.mean():.4f} m, std={L2_meas.std():.4f}, "
          f"min={L2_meas.min():.4f}, max={L2_meas.max():.4f}")
    print(f"L2 fixed:    {L2_fixed:.4f} m  (= {L2_fixed/0.0254:.2f} inches)")
    print(f"L1 ratio (fixed/measured): mean {L1_fixed/L1_meas.mean():.3f}")
    print(f"L2 ratio (fixed/measured): mean {L2_fixed/L2_meas.mean():.3f}")

    elb_rec = np.zeros_like(elb_true)
    wrs_rec_fixed = np.zeros_like(wrs_true)
    wrs_rec_meas = np.zeros_like(wrs_true)
    forearm_rec_unit = np.zeros_like(wrs_true)

    for i in range(n):
        q1, q2, q3, q4 = joint_angles.iloc[i][['q1', 'q2', 'q3', 'q4']]
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c3, s3 = np.cos(q3), np.sin(q3)
        c4, s4 = np.cos(q4), np.sin(q4)

        u = np.array([s2*c1, s2*s1, c2])
        e1 = np.array([c1*c2, s1*c2, -s2])
        e2 = np.array([-s1, c1, 0.0])

        elb_rec[i] = L1_fixed * u
        f_dir = -c4*u + s4*(c3*e1 + s3*e2)
        forearm_rec_unit[i] = f_dir
        wrs_rec_fixed[i] = elb_rec[i] + L2_fixed * f_dir
        wrs_rec_meas[i] = elb_rec[i] + L2_meas[i] * f_dir

    elb_err = np.linalg.norm(elb_rec - elb_true, axis=1)
    wrs_err_fixed = np.linalg.norm(wrs_rec_fixed - wrs_true, axis=1)
    wrs_err_meas = np.linalg.norm(wrs_rec_meas - wrs_true, axis=1)

    print("\n=== ELBOW vs WRIST RECONSTRUCTION ERRORS ===")
    print(f"Elbow err (fixed L1):           median {np.median(elb_err)*100:.2f} cm, max {elb_err.max()*100:.2f} cm")
    print(f"Wrist err (fixed L1, fixed L2): median {np.median(wrs_err_fixed)*100:.2f} cm, max {wrs_err_fixed.max()*100:.2f} cm")
    print(f"Wrist err (fixed L1, meas L2):  median {np.median(wrs_err_meas)*100:.2f} cm, max {wrs_err_meas.max()*100:.2f} cm")
    print("  ^ If 'meas L2' is much smaller than 'fixed L2', the issue is link-length mismatch.")
    print("    If they're similar, the issue is in the forearm DIRECTION (q3/q4 convention).")

    forearm_true_unit = forearm_true / L2_meas[:, None]
    cos_angle = np.clip(np.sum(forearm_rec_unit * forearm_true_unit, axis=1), -1, 1)
    angle_err_deg = np.degrees(np.arccos(cos_angle))

    print("\n=== FOREARM DIRECTION ERROR (the critical test) ===")
    print(f"Angle between reconstructed and true forearm direction:")
    print(f"  median {np.median(angle_err_deg):.2f}°, mean {angle_err_deg.mean():.2f}°, "
          f"max {angle_err_deg.max():.2f}°")
    print("  Interpretation:")
    print("    < 5°    : direction is correct, residual error is link-length only")
    print("    ~90°    : forearm direction is perpendicular (q3 or q4 convention off)")
    print("    ~180°   : forearm direction is FLIPPED (sign error on u, or q4 inverted)")
    print("    other   : more complex frame mismatch")

    wrs_rec_flip = elb_rec - L2_fixed * forearm_rec_unit
    wrs_err_flip = np.linalg.norm(wrs_rec_flip - wrs_true, axis=1)
    print(f"\n=== SIGN-FLIP TEST ===")
    print(f"If we negate the forearm direction:")
    print(f"  Wrist err: median {np.median(wrs_err_flip)*100:.2f} cm, max {wrs_err_flip.max()*100:.2f} cm")
    print("  If this is much smaller, the FK has a sign error on the forearm direction.")

    wrs_rec_alt = np.zeros_like(wrs_true)
    forearm_alt_unit = np.zeros_like(wrs_true)
    for i in range(n):
        q1, q2, q3, q4 = joint_angles.iloc[i][['q1', 'q2', 'q3', 'q4']]
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c3, s3 = np.cos(q3), np.sin(q3)
        c4, s4 = np.cos(q4), np.sin(q4)
        u = np.array([s2*c1, s2*s1, c2])
        e1 = np.array([c1*c2, s1*c2, -s2])
        e2 = np.array([-s1, c1, 0.0])
        f_dir_alt = c4*u + s4*(c3*e1 + s3*e2)
        forearm_alt_unit[i] = f_dir_alt
        wrs_rec_alt[i] = L1_fixed*u + L2_fixed * f_dir_alt
    cos_alt = np.clip(np.sum(forearm_alt_unit * forearm_true_unit, axis=1), -1, 1)
    angle_alt_deg = np.degrees(np.arccos(cos_alt))
    print(f"\n=== ALTERNATIVE CONVENTION (+u instead of -u) ===")
    print(f"Forearm angle error: median {np.median(angle_alt_deg):.2f}°, max {angle_alt_deg.max():.2f}°")

    print("\n=== IK INTERNAL CONSISTENCY (B1, B2, B3 vs forearm projections) ===")
    sample = min(5, n)
    for i in range(sample):
        q1, q2 = joint_angles.iloc[i][['q1', 'q2']]
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        u = np.array([s2*c1, s2*s1, c2])
        e1 = np.array([c1*c2, s1*c2, -s2])
        e2 = np.array([-s1, c1, 0.0])
        w = wrs_true[i]
        B1 = w @ e1
        B2 = w @ (-u)
        B3 = w @ e2
        q3_ik = np.arctan2(B3, B1)
        q4_ik = np.arctan2(np.sqrt(B1**2 + B3**2), B2 - L1_fixed)
        q3_actual, q4_actual = joint_angles.iloc[i][['q3', 'q4']]
        print(f"  Frame {i}: B=({B1:+.3f},{B2:+.3f},{B3:+.3f})  "
              f"q3 IK-derived={q3_ik:+.3f} actual={q3_actual:+.3f}  "
              f"q4 IK-derived={q4_ik:+.3f} actual={q4_actual:+.3f}")
