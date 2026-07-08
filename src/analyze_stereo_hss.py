"""
analyze_stereo_hss.py — Two-camera stereo HSS analysis with per-pitch images.

Usage:
    python3 analyze_stereo_hss.py <cam1_side.mov> <cam2_back.mov>
    python3 analyze_stereo_hss.py <cam1_side.mov> <cam2_back.mov> --skip-extract

What this does automatically:
    1. Audio cross-correlation  → sync offset between the two videos
    2. MediaPipe keypoint extraction on both videos
    3. Stereo self-calibration (Essential Matrix, USAC_MAGSAC)
    4. 3D HSS per detected pitch
    5. Per-pitch images saved to results/<cam1_stem>_<YYYY-MM-DD_HH-MM>/
         P1_hss.png      — frame of maximum HSS (just after foot plant)
         P1_release.png  — release frame (stride line + extension line)
         ...
         summary.txt

Results folder name includes date + time so every run is kept separately
(e.g. results/IMG_2425_2_2026-07-01_14-30/).
"""
import sys
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from find_audio_offset import find_offset
from extract_keypoints import extract
from run_calibration import (
    load_keypoints, match_frames, build_correspondences, calibrate,
    rotation_angle_deg, direction_angle_deg, nearest_frame_pair,
    detect_pitch_events, points_for_indices, compute_hss_series,
    plot_hss_over_time,
)
from geometry import (
    normalize_points, triangulate_points, resolve_scale,
    hip_shoulder_separation_angle,
    estimate_fov_from_video,
    FOV_DEGREES_CAM1, FOV_DEGREES_CAM2,
)

# ─────────────────────── CONFIG ─────────────────────────────────────────────
MODEL_PATH           = "pose_landmarker_heavy.task"
PERSON_HEIGHT_MM     = 1830.0
PLAYER_HEIGHT_M      = 1.83
PLAYER_WINGSPAN_M    = 1.93
HEIGHT_FRAME_TIME_SEC = 1.0
THROWS_RIGHT         = True
HSS_WINDOW_FRAMES    = 5
MIN_PITCH_ACTIVE_SEC = 0.20
MAX_PITCH_GAP_SEC    = 1.0    # 1.0s prevents one pitch from splitting into two
MIN_PEAK_ANG_VEL_DEG_S = 150.0
# starting guess for "is this HSS swing fast enough to be a real throw" -
# independent of the wrist-height heuristic used to find the segment.
# Print output shows the actual peak_ang_vel for every detected pitch;
# tune this number after looking at real vs. suspected-false-positive
# pitches from your own footage.
# ─────────────────────────────────────────────────────────────────────────────

LM = {
    "NOSE": 0,
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW":    13, "R_ELBOW":    14,
    "L_WRIST":    15, "R_WRIST":    16,
    "L_HIP":      23, "R_HIP":      24,
    "L_ANKLE":    27, "R_ANKLE":    28,
}
THROW_WRIST    = "R_WRIST"    if THROWS_RIGHT else "L_WRIST"
THROW_SHOULDER = "R_SHOULDER" if THROWS_RIGHT else "L_SHOULDER"
FRONT_ANKLE    = "L_ANKLE"    if THROWS_RIGHT else "R_ANKLE"
BACK_ANKLE     = "R_ANKLE"    if THROWS_RIGHT else "L_ANKLE"

# ─── colours (BGR) ───────────────────────────────────────────────────────────
COL_DOT       = (0, 220, 80)
COL_SHOULDER  = (0, 80, 220)
COL_HIP       = (0, 165, 255)
COL_ANKLE     = (0, 165, 255)
COL_EXTENSION = (0, 220, 255)
COL_WRIST     = (0, 80, 220)
COL_TEXT_BG   = (20, 20, 20)
COL_TEXT      = (255, 255, 255)
DOT_R = 8; LN = 3


# ═══════════════════════════════════════════════════════════════════════════════
# SIDE-VIEW helpers (for overlay images from cam1)
# ═══════════════════════════════════════════════════════════════════════════════
def vis(lm, thr=0.4):
    return lm.visibility > thr

def pt_lm(lms, name, w, h):
    lm = lms[LM[name]]
    return (int(lm.x * w), int(lm.y * h))

def dist_px(p1, p2):
    return float(np.linalg.norm(np.array(p1, float) - np.array(p2, float)))

def smooth_1d(arr, window=5):
    """
    Light moving-average smoothing applied before peak-picking.
    Without this, a single noisy/occluded frame (e.g. the throwing arm
    briefly self-occluding a shoulder/hip landmark, or motion blur at high
    arm speed) can produce one spurious spike that gets mistaken for the
    real HSS peak, even though it isn't the actual biomechanical moment.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")

def axis_angle_deg(p_left, p_right):
    dx = p_right[0] - p_left[0]; dy = p_right[1] - p_left[1]
    return float(np.degrees(np.arctan2(dy, dx)))

def arm_raised(lms):
    tw = lms[LM[THROW_WRIST]]; ts = lms[LM[THROW_SHOULDER]]
    return vis(tw) and vis(ts) and tw.y < ts.y

def height_based_scale_from_video(video_path):
    """Estimate pixel→metre scale from side-view video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2, min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35, min_tracking_confidence=0.35,
    )

    heights_px = []
    with mp_vision.PoseLandmarker.create_from_options(opts) as lmkr:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            ts_ms = int(idx * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lmkr.detect_for_video(img, ts_ms)
            if res.pose_landmarks:
                lms = min(res.pose_landmarks,
                          key=lambda l: (max(x.x for x in l)-min(x.x for x in l)) *
                                        (max(x.y for x in l)-min(x.y for x in l)))
                nose = lms[LM["NOSE"]]; la = lms[LM["L_ANKLE"]]; ra = lms[LM["R_ANKLE"]]
                if vis(nose) and vis(la) and vis(ra):
                    ankle_y = (la.y * h + ra.y * h) / 2
                    heights_px.append(abs(ankle_y - nose.y * h))
            idx += 1
    cap.release()
    if not heights_px:
        return 0.45 / 60, w, h
    return (PLAYER_HEIGHT_M / 1.05) / float(np.median(heights_px)), w, h

def read_frame_at(video_path, timestamp_sec):
    """Read a single frame from video at the given timestamp."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_sec * fps))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# ─── drawing ──────────────────────────────────────────────────────────────────
def draw_joints_from_json(frame, fr_json, fw, fh):
    for lm in fr_json["landmarks"]:
        if lm["visibility"] > 0.4:
            cv2.circle(frame, (int(lm["x"]), int(lm["y"])), DOT_R, COL_DOT, -1)

def label(frame, text, pos, fscale=0.65, thick=2):
    x, y = max(0, pos[0]), max(20, min(pos[1], frame.shape[0]-10))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
    cv2.rectangle(frame, (x-4, y-th-6), (x+tw+4, y+4), COL_TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fscale, COL_TEXT, thick)

def summary_card(frame, lines):
    fscale, thick = 0.7, 2; pad, lh = 10, 32
    max_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)[0][0] for l in lines)
    bh = pad*2 + lh*len(lines)
    cv2.rectangle(frame, (8,8), (8+max_w+pad*2, 8+bh), (20,20,20), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8+pad, 8+pad+lh*(i+1)),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, COL_TEXT, thick)

# ─── per-pitch image makers ───────────────────────────────────────────────────
def make_hss_image(cam1_video, fr1_json, pitch_n, hss_val):
    """
    HSS image: shot from the side-view camera (cam1).
    Shows shoulder axis (blue + box) and hip axis (orange).
    The frame chosen is the one where 3D HSS is maximum.
    """
    t = fr1_json["timestamp_sec"]
    frame = read_frame_at(cam1_video, t)
    if frame is None:
        return None
    fw, fh = frame.shape[1], frame.shape[0]
    canvas = frame.copy()
    draw_joints_from_json(canvas, fr1_json, fw, fh)

    lms = fr1_json["landmarks"]
    def lpt(name):
        lm = lms[LM[name]]
        return (int(lm["x"]), int(lm["y"]))

    ls, rs = lpt("L_SHOULDER"), lpt("R_SHOULDER")
    lh, rh = lpt("L_HIP"),      lpt("R_HIP")

    # shoulder axis + box
    cv2.line(canvas, ls, rs, COL_SHOULDER, LN)
    bx1 = (min(ls[0],rs[0])-6, min(ls[1],rs[1])-6)
    bx2 = (max(ls[0],rs[0])+6, max(ls[1],rs[1])+6)
    cv2.rectangle(canvas, bx1, bx2, COL_SHOULDER, 2)

    # hip axis
    cv2.line(canvas, lh, rh, COL_HIP, LN)

    # label between midpoints
    mid = ((ls[0]+rs[0])//2 + 10, (ls[1]+rs[1])//2 - 18)
    label(canvas, f"HSS: {hss_val:+.1f} deg", mid)

    summary_card(canvas, [
        f"Pitch {pitch_n} - Hip-Shoulder Separation (3D)",
        f"HSS  :  {hss_val:+.1f} deg",
        f"Time :  {t:.2f} s",
    ])
    return canvas

def make_release_image(cam1_video, fr1_json, pitch_n, scale, stride_m, ext_m):
    """
    Release image: shot from the side-view camera (cam1).
    Shows stride line (ankle-to-ankle) + extension line (rubber → wrist).
    """
    t = fr1_json["timestamp_sec"]
    frame = read_frame_at(cam1_video, t)
    if frame is None:
        return None
    fw, fh = frame.shape[1], frame.shape[0]
    canvas = frame.copy()
    draw_joints_from_json(canvas, fr1_json, fw, fh)

    lms = fr1_json["landmarks"]
    def lpt(name):
        lm = lms[LM[name]]
        return (int(lm["x"]), int(lm["y"]))

    wrist = lpt(THROW_WRIST)
    back  = lpt(BACK_ANKLE)
    front = lpt(FRONT_ANKLE)

    # stride line
    cv2.line(canvas, back, front, COL_ANKLE, LN)
    mid_ankle = ((back[0]+front[0])//2, max(back[1],front[1]) + 28)
    if stride_m:
        label(canvas, f"Stride: {stride_m:.2f} m", mid_ankle)

    # extension line (horizontal from back_ankle x to wrist x, at wrist height)
    ext_start = (back[0], wrist[1])
    cv2.line(canvas, ext_start, wrist, COL_EXTENSION, LN)
    cv2.circle(canvas, wrist, DOT_R+5, COL_WRIST, 2)
    if ext_m:
        label(canvas, f"Ext: {ext_m:.2f} m", (wrist[0]+12, wrist[1]-12))

    s_str = f"{stride_m:.2f} m" if stride_m else "--"
    e_str = f"{ext_m:.2f} m"   if ext_m    else "--"
    summary_card(canvas, [
        f"Pitch {pitch_n} - Release Point",
        f"Stride    :  {s_str}",
        f"Extension :  {e_str}",
        f"Time      :  {t:.2f} s",
    ])
    return canvas

# ─── stride / extension from cam1 JSON ───────────────────────────────────────
def pitching_direction_from_json(frames_json):
    xs = []
    for fr in frames_json:
        fa = fr["landmarks"][LM[FRONT_ANKLE]]
        ba = fr["landmarks"][LM[BACK_ANKLE]]
        if fa["visibility"] > 0.4 and ba["visibility"] > 0.4:
            xs.append(fa["x"] - ba["x"])
    return 1 if (np.median(xs) >= 0 if xs else True) else -1

def best_stride_and_release_from_json(pitch_frames_json, scale, direction):
    """
    From cam1 JSON frames for one pitch segment, find:
      - best release frame (wrist farthest forward)
      - stride length at that frame
      - extension at that frame
    """
    best_rel, best_val = None, -999
    for fr in pitch_frames_json:
        lms = fr["landmarks"]
        tw = lms[LM[THROW_WRIST]]; ts = lms[LM[THROW_SHOULDER]]
        if tw["visibility"] < 0.4 or ts["visibility"] < 0.4: continue
        if tw["y"] >= ts["y"]: continue   # arm not raised
        fwd = tw["x"] * direction
        if fwd > best_val:
            best_val = fwd
            best_rel = fr

    if best_rel is None:
        return None, None, None

    lms = best_rel["landmarks"]
    fa = lms[LM[FRONT_ANKLE]]; ba = lms[LM[BACK_ANKLE]]
    stride_m = None
    if fa["visibility"] > 0.4 and ba["visibility"] > 0.4:
        px = dist_px((fa["x"], fa["y"]), (ba["x"], ba["y"]))
        s = px * scale
        if 0.3 <= s / PLAYER_HEIGHT_M <= 1.2:
            stride_m = s

    tw = lms[LM[THROW_WRIST]]; ba = lms[LM[BACK_ANKLE]]
    ext_m = None
    if tw["visibility"] > 0.4 and ba["visibility"] > 0.4:
        dx_px = abs(tw["x"] - ba["x"])
        e = dx_px * scale * (0.5 + 0.5 * PLAYER_WINGSPAN_M / PLAYER_HEIGHT_M)
        if 0.3 * PLAYER_HEIGHT_M <= e <= 1.3 * PLAYER_HEIGHT_M:
            ext_m = e

    return best_rel, stride_m, ext_m


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("cam1", help="Side-view video (3rd-base line)")
    p.add_argument("cam2", help="Back-view video (behind catcher)")
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip MediaPipe extraction if JSON files already exist")
    return p.parse_args()

def step(n, total, msg):
    print(f"\n[{n}/{total}] {msg}")
    print("─" * 55)

def main():
    args = parse_args()
    cam1 = args.cam1
    cam2 = args.cam2

    # ── output folder: <cam1_stem>_<YYYY-MM-DD_HH-MM> ───────────────────────
    cam1_stem  = Path(cam1).stem
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir    = Path("results") / f"{cam1_stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {out_dir}/")

    kp1_path = str(out_dir / "keypoints_cam1.json")
    kp2_path = str(out_dir / "keypoints_cam2.json")

    # ── Step 1: audio offset ─────────────────────────────────────────────────
    step(1, 5, "Computing audio sync offset...")
    sync_override_path = Path("sync_offset.json")
    if sync_override_path.exists():
        override = json.loads(sync_override_path.read_text())
        offset_sec = override["offset_sec"]
        z_score = float("inf")
        print(f"  Using manually-verified offset from sync_offset.json: {offset_sec:+.4f} sec")
        print(f"  (delete sync_offset.json to go back to automatic audio-based sync)")
    else:
        offset_sec, z_score = find_offset(cam1, cam2)
        print(f"  Offset : {offset_sec:+.3f} sec")
        print(f"  Z-score: {z_score:.1f}  {'✓ trustworthy' if z_score > 5 else '⚠ low confidence'}")

    # ── Step 1b: FOV auto-detection ──────────────────────────────────────────
    step(2, 5, "Detecting camera FOV from video metadata...")
    fov1, src1 = estimate_fov_from_video(cam1, fallback_fov=FOV_DEGREES_CAM1)
    fov2, src2 = estimate_fov_from_video(cam2, fallback_fov=FOV_DEGREES_CAM2)
    print(f"  cam1 FOV: {fov1:.1f}°  ({src1})")
    print(f"  cam2 FOV: {fov2:.1f}°  ({src2})")

    # ── Step 3: keypoint extraction ──────────────────────────────────────────
    step(3, 5, "Extracting MediaPipe keypoints from both videos...")
    if args.skip_extract and Path(kp1_path).exists() and Path(kp2_path).exists():
        print("  --skip-extract: reusing existing JSON files.")
    else:
        print(f"  cam1 ({cam1}) ...")
        extract(cam1, kp1_path, model_path=MODEL_PATH)
        print(f"  cam2 ({cam2}) ...")
        extract(cam2, kp2_path, model_path=MODEL_PATH)

    # ── Step 4: stereo calibration + 3D HSS ─────────────────────────────────
    step(4, 5, "Stereo self-calibration + 3D HSS per pitch...")
    data1 = load_keypoints(kp1_path)
    data2 = load_keypoints(kp2_path)
    w1, h1 = data1["image_width"], data1["image_height"]
    w2, h2 = data2["image_width"], data2["image_height"]

    matches = match_frames(data1, data2, offset_sec)
    print(f"  Matched frames: {len(matches)} / {len(data1['frames'])}")
    if len(matches) == 0:
        print("  ERROR: 0 frames matched. Check audio offset / video overlap.")
        return

    landmark_indices = list(LM.values())
    pts1, pts2 = build_correspondences(matches, landmark_indices)
    R_full, T_full, _ = calibrate(pts1, pts2, w1, h1, w2, h2, fov1, fov2)

    # split-half consistency check
    # NOTE: interleaved (odd/even) split instead of chronological first-half /
    # second-half. A chronological split can put most of the actual pitching
    # motion (the only frames with real parallax/depth signal) almost entirely
    # into one half, starving the other half's Essential Matrix estimate and
    # making the two R's disagree even when calibration itself is fine.
    # Interleaving spreads pitch vs. idle frames evenly across both halves.
    pts1a, pts2a = build_correspondences(matches[0::2], landmark_indices)
    pts1b, pts2b = build_correspondences(matches[1::2], landmark_indices)
    Ra, Ta, _ = calibrate(pts1a, pts2a, w1, h1, w2, h2, fov1, fov2)
    Rb, Tb, _ = calibrate(pts1b, pts2b, w1, h1, w2, h2, fov1, fov2)
    rot_d = rotation_angle_deg(Ra, Rb)
    dir_d = direction_angle_deg(Ta, Tb)
    consistent = bool(rot_d <= 5 and dir_d <= 5)

    # scale
    fr1h, fr2h, _ = nearest_frame_pair(matches, HEIGHT_FRAME_TIME_SEC)
    hi = [LM["NOSE"], LM["L_ANKLE"]]
    pts1h = points_for_indices(fr1h, hi)
    pts2h = points_for_indices(fr2h, hi)
    norm1h = normalize_points(pts1h, w1, h1, fov1)
    norm2h = normalize_points(pts2h, w2, h2, fov2)
    pts3d_h = triangulate_points(R_full, T_full, norm1h, norm2h)
    scale_factor, _ = resolve_scale(pts3d_h, 0, 1, PERSON_HEIGHT_MM)
    baseline_mm = float(np.linalg.norm(T_full.ravel() * scale_factor))

    # HSS time series + pitch events
    all_times, all_angles = compute_hss_series(
        matches, R_full, T_full, scale_factor, w1, h1, w2, h2, fov1, fov2)
    events = detect_pitch_events(
        matches, MIN_PITCH_ACTIVE_SEC, MAX_PITCH_GAP_SEC)
    print(f"  Pitches detected: {len(events)}")

    half_w = HSS_WINDOW_FRAMES // 2
    pitches_info = []
    for i, (s, e) in enumerate(events, 1):
        seg_angles = all_angles[s:e+1]
        # smooth before peak-picking so one noisy/occluded frame can't
        # hijack which moment gets reported as "the" HSS peak
        smoothed = smooth_1d(seg_angles, window=5)
        peak_local = int(np.argmax(np.abs(smoothed)))
        peak_idx   = s + peak_local
        lo = max(0, peak_idx - half_w)
        hi_idx = min(len(matches), peak_idx + half_w + 1)
        mean_hss = float(all_angles[lo:hi_idx].mean())
        std_hss  = float(all_angles[lo:hi_idx].std())
        pitches_info.append({
            "n": i,
            "peak_match_idx": peak_idx,
            "peak_time_sec": float(all_times[peak_idx]),
            "mean_deg": round(mean_hss, 1),
            "std_deg":  round(std_hss, 1),
            "seg_start": s, "seg_end": e,
        })

    # plot HSS over time
    plot_windows = [{"lo_t": all_times[max(0, p["peak_match_idx"]-half_w)],
                     "hi_t": all_times[min(len(matches)-1, p["peak_match_idx"]+half_w)],
                     "mean_deg": p["mean_deg"], "label": f"P{p['n']}"}
                    for p in pitches_info]
    plot_hss_over_time(all_times, all_angles, plot_windows,
                       str(out_dir / "hss_over_time.png"))

    # ── save frame-by-frame 3D HSS timeseries for overlay video ─────────────
    # pitch_visualizer.py reads this to show live 3D HSS on each frame
    hss_ts = {
        "times": all_times.tolist(),
        "angles_deg": all_angles.tolist(),
    }
    with open(out_dir / "hss_timeseries.json", "w") as f:
        json.dump(hss_ts, f)
    print(f"  HSS timeseries saved ({len(all_times)} frames) → hss_timeseries.json")

    # ── save matched (cam1, cam2) frame pairs for the dual-camera overlay ───
    # video (pitch_visualizer.py draws both camera views side by side using
    # this, so it doesn't have to re-run match_frames() itself)
    matches_export = [{
        "t1": m[0]["timestamp_sec"], "lm1": m[0]["landmarks"],
        "t2": m[1]["timestamp_sec"], "lm2": m[1]["landmarks"],
    } for m in matches]
    with open(out_dir / "matches.json", "w") as f:
        json.dump(matches_export, f)
    print(f"  Matched frame pairs saved ({len(matches_export)}) → matches.json")

    # ── Step 5: per-pitch images ─────────────────────────────────────────────
    step(5, 5, "Saving per-pitch images...")
    # pixel scale from cam1 for stride/extension
    px_scale, _, _ = height_based_scale_from_video(cam1)
    direction = pitching_direction_from_json(data1["frames"])

    summary_lines = [
        "Pitch  |  HSS (3D, deg)  |  Stride (m)  |  Extension (m)",
        "-" * 58,
    ]
    pitches_export = []

    for p in pitches_info:
        pitch_n = p["n"]

        # HSS's own rotational dynamics within this segment - independent of
        # the wrist-height heuristic used to find the segment in the first
        # place. A real throw whips the hips/shoulders apart quickly; a slow
        # arm-raise (adjusting glove, catching a return throw) won't.
        seg_angles_raw = all_angles[p["seg_start"]:p["seg_end"] + 1]
        seg_times_raw  = all_times[p["seg_start"]:p["seg_end"] + 1]
        if len(seg_times_raw) > 1:
            d_ang = np.diff(seg_angles_raw)
            d_t   = np.diff(seg_times_raw)
            peak_ang_vel = float(np.abs(d_ang / np.where(d_t == 0, 1e-6, d_t)).max())
        else:
            peak_ang_vel = 0.0

        # ── HSS image: use the cam1 frame at peak HSS time ──────────────────
        peak_fr1 = matches[p["peak_match_idx"]][0]
        img_hss = make_hss_image(cam1, peak_fr1, pitch_n, p["mean_deg"])
        if img_hss is not None:
            cv2.imwrite(str(out_dir / f"P{pitch_n}_hss.png"), img_hss)
            print(f"  P{pitch_n}_hss.png      HSS = {p['mean_deg']:+.1f}°  "
                  f"(±{p['std_deg']})  peak_ang_vel={peak_ang_vel:.0f} deg/s")

        # ── release image: find best release frame in seg from cam1 JSON ────
        seg_frames_json = [matches[i][0] for i in range(p["seg_start"], p["seg_end"]+1)]
        rel_fr, stride_m, ext_m = best_stride_and_release_from_json(
            seg_frames_json, px_scale, direction)
        release_time_sec = rel_fr["timestamp_sec"] if rel_fr is not None else None
        if rel_fr is not None:
            img_rel = make_release_image(cam1, rel_fr, pitch_n, px_scale, stride_m, ext_m)
            if img_rel is not None:
                cv2.imwrite(str(out_dir / f"P{pitch_n}_release.png"), img_rel)
                s_str = f"{stride_m:.2f}" if stride_m else "--"
                e_str = f"{ext_m:.2f}"   if ext_m    else "--"
                print(f"  P{pitch_n}_release.png  stride={s_str}m  ext={e_str}m")

        h_str = f"{p['mean_deg']:+.1f}"
        s_str = f"{stride_m:.2f}" if stride_m is not None else "--"
        e_str = f"{ext_m:.2f}"   if ext_m    is not None else "--"

        no_stride_ext = (stride_m is None and ext_m is None)
        slow_rotation = peak_ang_vel < MIN_PEAK_ANG_VEL_DEG_S
        reasons = []
        if no_stride_ext: reasons.append("no stride/ext")
        if slow_rotation: reasons.append(f"slow rotation {peak_ang_vel:.0f} deg/s")
        flag = f"  (rough estimate - {', '.join(reasons)})" if reasons else ""

        summary_lines.append(
            f"P{pitch_n:3d}   |  {h_str:>14}  |  {s_str:>11}  |  {e_str:>13}{flag}")

        pitches_export.append({
            "n": pitch_n,
            "start_sec": float(all_times[p["seg_start"]]),
            "end_sec": float(all_times[p["seg_end"]]),
            "peak_time_sec": p["peak_time_sec"],
            "release_time_sec": release_time_sec,
            "mean_deg": p["mean_deg"],
            "std_deg": p["std_deg"],
            "peak_ang_vel": peak_ang_vel,
        })

    with open(out_dir / "pitches.json", "w") as f:
        json.dump(pitches_export, f, indent=2)
    print(f"  Pitch segments saved ({len(pitches_export)} pitches) → pitches.json")

    summary_lines += [
        "",
        f"Camera baseline    : {baseline_mm:.0f} mm",
        f"Consistency check  : {'OK' if consistent else 'CHECK'} "
        f"(rot {rot_d:.1f} deg, dir {dir_d:.1f} deg)",
        f"Matched frames     : {len(matches)} / {len(data1['frames'])}",
        f"Audio offset       : {offset_sec:+.3f} sec",
        f"cam1 FOV           : {fov1:.1f} deg  ({src1})",
        f"cam2 FOV           : {fov2:.1f} deg  ({src2})",
    ]

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\n{'='*55}")
    print(f"  DONE — {len(pitches_info)} pitch(es), "
          f"{len(pitches_info)*2} images saved")
    print(f"  Folder : {out_dir}/")
    print(f"  Summary: {summary_path}")
    print(f"{'='*55}\n")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()