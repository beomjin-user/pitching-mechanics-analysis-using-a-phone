"""
run_calibration.py — Full orchestration of the self-calibration pipeline.

Steps:
  1. Load two keypoint JSON files (from extract_keypoints.py).
  2. Apply the audio offset (from find_audio_offset.py) to align frame
     timestamps between the two cameras.
  3. Match frames by nearest timestamp.
  4. Build 2D correspondences from matched frames using a fixed set of
     body landmarks (shoulders, hips, ankles, nose).
  5. Run geometry.py: normalize -> estimate E -> recover R,T -> triangulate.
  6. Resolve scale using a known height at a frame where the pitcher is
     standing upright.
  7. SPLIT-HALF CONSISTENCY CHECK: redo steps 5-6 independently on the
     first half and second half of matched frames. Without a checkerboard
     there's no ground truth to check against directly, so agreement
     between the two halves is the best confidence signal available.
  8. AUTOMATICALLY DETECT EVERY PITCH in the clip (same "throwing wrist
     above throwing shoulder" heuristic pitch_summary_v2.py already uses
     to filter out non-pitching frames) and report HSS for each one
     separately -- not just a single timestamp you pick by hand.

EDIT THE CONFIG SECTION BELOW before running.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import (
    normalize_points, estimate_essential_matrix, recover_pose,
    triangulate_points, resolve_scale, hip_shoulder_separation_angle,
    FOV_DEGREES_CAM1, FOV_DEGREES_CAM2,
)

# ----------------- CONFIG (EDIT THESE) -----------------
KEYPOINTS_JSON_CAM1 = "keypoints_cam1.json"   # side view (3rd-base line), from extract_keypoints.py
KEYPOINTS_JSON_CAM2 = "keypoints_cam2.json"   # back view (behind catcher), from extract_keypoints.py

# offset_sec from find_audio_offset.py, where cam2_timestamp = cam1_timestamp + offset_sec.
# find_audio_offset.py prints the exact line to paste here — use it as-is.
AUDIO_OFFSET_SEC = -5.307

PERSON_HEIGHT_MM = 1830.0          # EDIT: your actual height in mm
HEIGHT_FRAME_TIME_SEC = 1     # EDIT: a cam1 timestamp where you're standing upright (pre-windup, any single pitch is fine)

HSS_WINDOW_FRAMES = 5              # how many consecutive matched frames to average around each pitch's peak HSS
                                    # (single-frame HSS is noise-dominated — see geometry.py docstring)
MIN_PITCH_ACTIVE_SEC = 0.15        # minimum duration of "throwing wrist above shoulder" to count as a real pitch
                                    # (filters out brief hand-raises / detection blips)
MAX_PITCH_GAP_SEC = 0.3            # merge brief tracking dropouts within the same pitch motion
                                    # (too small = one pitch gets split into two; too large = two pitches merge into one)
# ---------------------------------------------------------

# MediaPipe Pose landmark indices (BlazePose 33-point topology)
LM = {
    "NOSE": 0,
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW": 13, "R_ELBOW": 14,
    "L_WRIST": 15, "R_WRIST": 16,
    "L_HIP": 23, "R_HIP": 24,
    "L_ANKLE": 27, "R_ANKLE": 28,
}


def load_keypoints(path):
    with open(path) as f:
        return json.load(f)


def match_frames(data1, data2, offset_sec, max_dt=1.0 / 60):
    """Match cam1 frames to the nearest cam2 frame.

    offset_sec convention (from find_audio_offset.py):
        cam2_timestamp = cam1_timestamp + offset_sec
    """
    frames2 = data2["frames"]
    times2 = np.array([fr["timestamp_sec"] for fr in frames2])

    matches = []
    for fr1 in data1["frames"]:
        t1_in_cam2_clock = fr1["timestamp_sec"] + offset_sec
        idx = int(np.argmin(np.abs(times2 - t1_in_cam2_clock)))
        dt = abs(times2[idx] - t1_in_cam2_clock)
        if dt <= max_dt:
            matches.append((fr1, frames2[idx], dt))
    return matches


def build_correspondences(matches, landmark_indices):
    pts1, pts2 = [], []
    for fr1, fr2, _dt in matches:
        for idx in landmark_indices:
            lm1 = fr1["landmarks"][idx]
            lm2 = fr2["landmarks"][idx]
            if lm1["visibility"] < 0.5 or lm2["visibility"] < 0.5:
                continue
            pts1.append([lm1["x"], lm1["y"]])
            pts2.append([lm2["x"], lm2["y"]])
    return np.array(pts1), np.array(pts2)


def calibrate(pts1_px, pts2_px, w1, h1, w2, h2,
              fov1=None, fov2=None):
    f1 = fov1 if fov1 is not None else FOV_DEGREES_CAM1
    f2 = fov2 if fov2 is not None else FOV_DEGREES_CAM2
    norm1 = normalize_points(pts1_px, w1, h1, f1)
    norm2 = normalize_points(pts2_px, w2, h2, f2)
    E, mask = estimate_essential_matrix(norm1, norm2)
    R, T, mask_pose = recover_pose(E, norm1, norm2, mask)
    inliers = mask_pose.ravel() > 0
    pts3d = triangulate_points(R, T, norm1[inliers], norm2[inliers])
    return R, T, pts3d


def rotation_angle_deg(R1, R2):
    R_rel = R1.T @ R2
    cos_a = np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def direction_angle_deg(T1, T2):
    t1 = T1.ravel() / np.linalg.norm(T1)
    t2 = T2.ravel() / np.linalg.norm(T2)
    cos_a = np.clip(np.dot(t1, t2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def nearest_frame_pair(matches, target_time_sec):
    return min(matches, key=lambda m: abs(m[0]["timestamp_sec"] - target_time_sec))


def detect_pitch_events(matches, min_active_sec=MIN_PITCH_ACTIVE_SEC, max_gap_sec=MAX_PITCH_GAP_SEC):
    """
    Auto-detect every distinct pitch in the clip using the same heuristic
    pitch_summary_v2.py already uses to exclude static/non-pitching frames:
    the throwing wrist rising above the throwing shoulder. Assumes a
    right-handed pitcher (R_WRIST vs R_SHOULDER) -- swap to L_WRIST/
    L_SHOULDER below if left-handed.

    Returns a list of (start_idx, end_idx) index pairs into `matches`,
    one per detected pitch, in time order.
    """
    active = []
    for fr1, _fr2, _dt in matches:
        wrist = fr1["landmarks"][LM["R_WRIST"]]
        shoulder = fr1["landmarks"][LM["R_SHOULDER"]]
        is_active = (
            wrist["visibility"] > 0.5 and shoulder["visibility"] > 0.5
            and wrist["y"] < shoulder["y"]  # smaller y = higher in the image = "above"
        )
        active.append(is_active)

    times = np.array([m[0]["timestamp_sec"] for m in matches])

    raw_segments = []
    seg_start = None
    for i, is_active in enumerate(active):
        if is_active:
            if seg_start is None:
                seg_start = i
        else:
            if seg_start is not None:
                raw_segments.append((seg_start, i - 1))
                seg_start = None
    if seg_start is not None:
        raw_segments.append((seg_start, len(active) - 1))

    if not raw_segments:
        return []

    merged = [raw_segments[0]]
    for seg in raw_segments[1:]:
        if times[seg[0]] - times[merged[-1][1]] <= max_gap_sec:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    return [seg for seg in merged if times[seg[1]] - times[seg[0]] >= min_active_sec]


def points_for_indices(frame, indices):
    return np.array([[frame["landmarks"][i]["x"], frame["landmarks"][i]["y"]] for i in indices])


def compute_hss_series(matches, R_full, T_full, scale_factor,
                        w1, h1, w2, h2, fov1=None, fov2=None):
    """Compute the 3D HSS angle at every matched frame (for plotting over time)."""
    f1 = fov1 if fov1 is not None else FOV_DEGREES_CAM1
    f2 = fov2 if fov2 is not None else FOV_DEGREES_CAM2
    hss_idx = [LM["L_SHOULDER"], LM["R_SHOULDER"], LM["L_HIP"], LM["R_HIP"]]
    local_idx = {"L_SHOULDER": 0, "R_SHOULDER": 1, "L_HIP": 2, "R_HIP": 3}
    times, angles = [], []
    for fr1, fr2, _dt in matches:
        pts1 = points_for_indices(fr1, hss_idx)
        pts2 = points_for_indices(fr2, hss_idx)
        norm1 = normalize_points(pts1, w1, h1, f1)
        norm2 = normalize_points(pts2, w2, h2, f2)
        pts3d = triangulate_points(R_full, T_full, norm1, norm2) * scale_factor
        times.append(fr1["timestamp_sec"])
        angles.append(hip_shoulder_separation_angle(pts3d, local_idx))
    return np.array(times), np.array(angles)


def plot_hss_over_time(times, angles, pitch_windows, out_path):
    """Save a plot of HSS angle vs. time, highlighting the averaging window
    used for each detected pitch's headline number -- so it's visually
    clear WHEN in the clip each pitch's separation was measured.

    pitch_windows: list of dicts with keys 'lo_t', 'hi_t', 'mean_deg', 'label'
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(times, angles, color="#94a3b8", linewidth=1.2, marker="o", markersize=2.5, alpha=0.7,
            label="HSS (all frames, noisy)")

    colors = ["#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#0ea5e9", "#ec4899"]
    for i, w in enumerate(pitch_windows):
        color = colors[i % len(colors)]
        ax.axvspan(w["lo_t"], w["hi_t"], color=color, alpha=0.25)
        ax.text((w["lo_t"] + w["hi_t"]) / 2, ax.get_ylim()[1], w["label"],
                ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
        ax.plot([w["lo_t"], w["hi_t"]], [w["mean_deg"], w["mean_deg"]],
                color=color, linewidth=2.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hip-Shoulder Separation (deg)")
    ax.set_title("3D Hip-Shoulder Separation Over Time (all detected pitches)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    data1 = load_keypoints(KEYPOINTS_JSON_CAM1)
    data2 = load_keypoints(KEYPOINTS_JSON_CAM2)
    w1, h1 = data1["image_width"], data1["image_height"]
    w2, h2 = data2["image_width"], data2["image_height"]

    matches = match_frames(data1, data2, AUDIO_OFFSET_SEC)
    if len(matches) == 0:
        print(f"Matched 0 frame pairs (of {len(data1['frames'])} cam1 frames with detected pose).")
        print("Calibration cannot proceed. Check AUDIO_OFFSET_SEC and that both videos overlap in time.")
        return

    landmark_indices = list(LM.values())

    # ---- full estimate ----
    pts1, pts2 = build_correspondences(matches, landmark_indices)
    R_full, T_full, pts3d_full = calibrate(pts1, pts2, w1, h1, w2, h2)

    # ---- split-half consistency check ----
    mid = len(matches) // 2
    first_half, second_half = matches[:mid], matches[mid:]
    pts1a, pts2a = build_correspondences(first_half, landmark_indices)
    pts1b, pts2b = build_correspondences(second_half, landmark_indices)
    R_a, T_a, _ = calibrate(pts1a, pts2a, w1, h1, w2, h2)
    R_b, T_b, _ = calibrate(pts1b, pts2b, w1, h1, w2, h2)
    rot_disagreement = rotation_angle_deg(R_a, R_b)
    dir_disagreement = direction_angle_deg(T_a, T_b)
    consistent = bool(rot_disagreement <= 5 and dir_disagreement <= 5)

    # ---- resolve scale using known height ----
    fr1h, fr2h, _ = nearest_frame_pair(matches, HEIGHT_FRAME_TIME_SEC)
    height_idx = [LM["NOSE"], LM["L_ANKLE"]]
    pts1h = points_for_indices(fr1h, height_idx)
    pts2h = points_for_indices(fr2h, height_idx)
    norm1h = normalize_points(pts1h, w1, h1, FOV_DEGREES_CAM1)
    norm2h = normalize_points(pts2h, w2, h2, FOV_DEGREES_CAM2)
    pts3d_h = triangulate_points(R_full, T_full, norm1h, norm2h)
    scale_factor, _ = resolve_scale(pts3d_h, idx_top=0, idx_bottom=1, known_distance_mm=PERSON_HEIGHT_MM)
    T_full_mm = T_full.ravel() * scale_factor
    baseline_mm = float(np.linalg.norm(T_full_mm))

    # ---- HSS: full time series (for the plot), then per-pitch peak+window ----
    all_times, all_angles = compute_hss_series(matches, R_full, T_full, scale_factor, w1, h1, w2, h2)

    events = detect_pitch_events(matches)
    half_window = HSS_WINDOW_FRAMES // 2

    pitches = []
    for i, (s, e) in enumerate(events, start=1):
        seg_angles = all_angles[s:e + 1]
        peak_local_idx = int(np.argmax(np.abs(seg_angles)))
        peak_idx = s + peak_local_idx
        lo = max(0, peak_idx - half_window)
        hi = min(len(matches), peak_idx + half_window + 1)
        frame_angles = all_angles[lo:hi]
        pitches.append({
            "n": i,
            "peak_time_sec": round(float(all_times[peak_idx]), 2),
            "window_lo_t": float(all_times[lo]),
            "window_hi_t": float(all_times[hi - 1]),
            "mean_deg": round(float(frame_angles.mean()), 1),
            "std_deg": round(float(frame_angles.std()), 1),
            "n_frames": len(frame_angles),
            "high_spread": bool(frame_angles.std() > 10),
        })

    # ---------------- write clean results files ----------------
    out_dir = Path("results") / "self_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_path = out_dir / "hip_shoulder_separation.png"
    plot_windows = [
        {"lo_t": p["window_lo_t"], "hi_t": p["window_hi_t"], "mean_deg": p["mean_deg"], "label": f"P{p['n']}"}
        for p in pitches
    ]
    plot_hss_over_time(all_times, all_angles, plot_windows, plot_path)

    results = {
        "matched_frames": len(matches),
        "total_cam1_frames": len(data1["frames"]),
        "low_match_warning": len(matches) < 30,
        "split_half_consistency": {
            "rotation_disagreement_deg": round(float(rot_disagreement), 2),
            "direction_disagreement_deg": round(float(dir_disagreement), 2),
            "consistent": consistent,
        },
        "scale_factor": round(float(scale_factor), 4),
        "camera_baseline_mm": round(baseline_mm, 1),
        "pitches_detected": len(pitches),
        "pitches": pitches,
    }
    with open(out_dir / "calibration_result.json", "w") as f:
        json.dump(results, f, indent=2)

    warnings = []
    if len(matches) < 30:
        warnings.append(f"few matched frames ({len(matches)}) — check AUDIO_OFFSET_SEC / video overlap")
    if not consistent:
        warnings.append(f"split-half check inconsistent (rot {rot_disagreement:.1f}, dir {dir_disagreement:.1f} deg)")
    if not pitches:
        warnings.append("no pitches auto-detected — check MIN_PITCH_ACTIVE_SEC/MAX_PITCH_GAP_SEC, "
                         "or that the throwing arm (right wrist/shoulder) is visible in cam1")
    for p in pitches:
        if p["high_spread"]:
            warnings.append(f"Pitch {p['n']}: HSS spread is high (std {p['std_deg']} deg) — treat as rough")

    lines = [
        f"Camera baseline    : {baseline_mm:.0f} mm",
        f"Consistency check  : {'OK' if consistent else 'CHECK'} "
        f"(rot {rot_disagreement:.1f} deg, dir {dir_disagreement:.1f} deg)",
        f"Matched frames     : {len(matches)} / {len(data1['frames'])}",
        f"Pitches detected   : {len(pitches)}",
        "",
    ]
    for p in pitches:
        lines.append(f"Pitch {p['n']} (t={p['peak_time_sec']}s): "
                      f"Hip-Shoulder Separation = {p['mean_deg']:+.1f} deg  "
                      f"(± {p['std_deg']}, n={p['n_frames']})")
    lines.append("")
    lines.append(f"See {plot_path.name} for HSS over the full clip, all pitches marked.")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")

    summary_path = out_dir / "calibration_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Done. {len(pitches)} pitch(es) detected.")
    print(f"Results: {summary_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
