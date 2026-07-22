"""
Pitching Mechanics Analysis — version with automatic motion-phase classification
Output: Stride Length, Release Extension, Max Hip-Shoulder Separation (measured at release)

Filming guide:
    You only need one side-view video (3rd-base or 1st-base side, 90 degrees
    to the pitching direction).
    This angle is optimal not just for stride/extension but also for
    hip-shoulder separation measurement.
    (Filming from the front/back makes the hip/shoulder rotation axis nearly
     parallel to the camera, so the rotation angle measures smaller than it
     actually is — always film from the side.)

Key improvement:
    The previous version searched the whole video for "the max value within
    a normal range," so it could pick up values from motions unrelated to
    the actual pitch, like lifting the leg during the windup.
    This version analyzes wrist speed/position patterns to automatically
    classify every frame into one of six phases, and only measures HSS on
    frames classified as the "release" phase.
    Stride/extension are only measured on frames where the throwing hand is
    above the shoulder, which prevents a static ready stance from being
    mistaken for foot plant/release.

Six pitching phases:
    1. windup          — the leg-lift preparation motion
    2. stride           — the lead foot driving toward the target
    3. cocking          — right after foot plant, arm still cocked back
    4. acceleration     — the window where wrist speed rises sharply
    5. release          — the instant of peak wrist speed (ball release)
    6. follow_through   — the deceleration phase after release

Required packages:
    pip install mediapipe opencv-python numpy pandas matplotlib
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# ──────────────────────────────────────────
# CONFIG — this is the only section you should need to edit
# ──────────────────────────────────────────
# A single side-view video (3rd-base or 1st-base side, 90 degrees) is
# enough to measure HSS, stride, and extension all at once.
# This angle is optimal for all three metrics (hip/shoulder rotation reads
# perpendicular to the camera), so there's no need for a separate back-view
# video anymore.
#
# Instead of hardcoding VIDEO_PATH, video files in the current folder are
# auto-discovered and offered as a numbered list each run (see
# select_video_interactively below). Just drop a new video into this folder
# — no need to open the code and edit a filename.
VIDEO_SEARCH_DIR = "."                   # folder to search for videos (default: current folder)
VIDEO_EXTENSIONS  = [".mov", ".mp4", ".MOV", ".MP4"]

MODEL_PATH       = "pose_landmarker_heavy.task"
PLAYER_HEIGHT_M  = 1.83                  # your height (meters)
PLAYER_WINGSPAN_M = 1.93                 # your wingspan (fingertip-to-fingertip, arms spread, meters)
THROWS_RIGHT     = True                  # True = right-handed pitcher, False = left-handed

# Set to 2+ if the frame includes anyone besides the pitcher (catcher,
# observer, etc). Especially needed for a catch-play framing where someone
# stands right in front of the camera.
NUM_POSES         = 2
# "smallest": selects whoever appears smallest on screen (= farthest from
#   the camera) as the pitcher. Fits the common catch-play framing where
#   the person nearer the camera (not the pitcher) appears larger.
PITCHER_SELECT_MODE = "smallest"

# Outlier thresholds (kept as a final safety net, separate from phase classification)
STRIDE_RATIO_MIN, STRIDE_RATIO_MAX = 0.3, 1.2
HSS_MIN, HSS_MAX = 0, 90

IDX = {
    "nose":       0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow":    13, "r_elbow":    14,
    "l_wrist":    15, "r_wrist":    16,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
}

PHASES = ["windup", "stride", "cocking", "acceleration", "release", "follow_through"]
PHASE_COLORS_BGR = {
    "windup":         (180, 180, 180),
    "stride":         (255, 165, 0),
    "cocking":        (0, 200, 255),
    "acceleration":   (60, 60, 230),
    "release":        (0, 255, 0),
    "follow_through": (200, 100, 200),
}
PHASE_COLORS_HEX = {
    "windup":         "#888780",
    "stride":         "#0F6E56",
    "cocking":        "#BA7517",
    "acceleration":   "#D85A30",
    "release":        "#1D9E75",
    "follow_through": "#534AB7",
}


# ──────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────
def lm_to_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h])


def axis_angle(p_left, p_right):
    dx = p_right[0] - p_left[0]
    dy = p_right[1] - p_left[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def dist(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def create_landmarker(model_path, mode, num_poses=1):
    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mode,
        num_poses=num_poses,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return PoseLandmarker.create_from_options(options)


def select_pitcher_pose(pose_landmarks_list, mode="smallest"):
    """
    Automatically selects the pitcher when multiple people are detected.

    mode="smallest": treats whoever has the smallest shoulder width
                      (= appears smallest on screen, i.e. farthest from
                      the camera) as the pitcher.
                      In a catch-play framing where someone else (catcher/
                      observer) stands right in front of the camera, the
                      pitcher is usually farther away and appears smaller.

    Returns: the selected person's landmark list, or None if nobody was detected
    """
    if not pose_landmarks_list:
        return None
    if len(pose_landmarks_list) == 1:
        return pose_landmarks_list[0]

    if mode == "smallest":
        best = None
        best_size = None
        for lms in pose_landmarks_list:
            try:
                l_sh = lms[11]
                r_sh = lms[12]
                w = abs(l_sh.x - r_sh.x)
                h = abs(l_sh.y - r_sh.y)
                size = (w**2 + h**2) ** 0.5
            except (IndexError, AttributeError):
                continue
            if best_size is None or size < best_size:
                best_size = size
                best = lms
        return best if best is not None else pose_landmarks_list[0]

    return pose_landmarks_list[0]


# ──────────────────────────────────────────
# Extract every joint coordinate up front (run once, reuse everywhere)
# ──────────────────────────────────────────
def extract_all_landmarks(video_path, model_path, throws_right, num_poses=1, pitcher_select="smallest"):
    """
    Extracts the needed joint coordinates from every frame of the video and
    returns them as a DataFrame. Phase classification, HSS, and stride are
    all computed downstream from this single DataFrame.

    Setting num_poses > 1 detects everyone in frame and then picks just the
    pitcher via select_pitcher_pose(). Use this for footage where someone
    else (catcher/observer) is also in frame, like catch-play video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_factor = 3 if w > 1500 else 1
    rows = []

    throw_wrist = "r_wrist" if throws_right else "l_wrist"
    throw_elbow = "r_elbow" if throws_right else "l_elbow"
    throw_shoulder = "r_shoulder" if throws_right else "l_shoulder"
    front_ankle = "l_ankle" if throws_right else "r_ankle"
    back_ankle  = "r_ankle" if throws_right else "l_ankle"
    lead_knee   = "l_knee" if throws_right else "r_knee"

    with create_landmarker(model_path, RunningMode.VIDEO, num_poses=num_poses) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            proc = cv2.resize(frame, (w // scale_factor, h // scale_factor)) if scale_factor > 1 else frame
            ph, pw = proc.shape[:2]

            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            row = {"frame": frame_idx, "time_sec": frame_idx / fps, "detected": False}

            lms = select_pitcher_pose(result.pose_landmarks, mode=pitcher_select) if result.pose_landmarks else None

            if lms is not None:
                avg_vis = np.mean([lms[i].visibility for i in IDX.values()])

                if avg_vis > 0.3:
                    row["detected"] = True
                    row["avg_vis"] = avg_vis
                    for name, idx in IDX.items():
                        xy = lm_to_xy(lms[idx], pw, ph)
                        row[f"{name}_x"] = xy[0]
                        row[f"{name}_y"] = xy[1]
                        row[f"{name}_vis"] = lms[idx].visibility

                    row["shoulder_w_px"] = dist(
                        lm_to_xy(lms[IDX["l_shoulder"]], pw, ph),
                        lm_to_xy(lms[IDX["r_shoulder"]], pw, ph),
                    )
                    # also stash the key classification variables under standard names
                    row["throw_wrist_x"] = lm_to_xy(lms[IDX[throw_wrist]], pw, ph)[0]
                    row["throw_wrist_y"] = lm_to_xy(lms[IDX[throw_wrist]], pw, ph)[1]
                    row["throw_shoulder_y"] = lm_to_xy(lms[IDX[throw_shoulder]], pw, ph)[1]
                    row["lead_knee_y"]   = lm_to_xy(lms[IDX[lead_knee]], pw, ph)[1]
                    row["front_ankle_x"] = lm_to_xy(lms[IDX[front_ankle]], pw, ph)[0]
                    row["front_ankle_y"] = lm_to_xy(lms[IDX[front_ankle]], pw, ph)[1]
                    row["back_ankle_x"]  = lm_to_xy(lms[IDX[back_ankle]], pw, ph)[0]
                    row["back_ankle_y"]  = lm_to_xy(lms[IDX[back_ankle]], pw, ph)[1]
                    row["throw_wrist_vis"] = lms[IDX[throw_wrist]].visibility

            rows.append(row)
            frame_idx += 1

    cap.release()
    df = pd.DataFrame(rows)
    print(f"  video: {w}x{h} / {fps:.1f}fps / {total} frames, detected {df['detected'].sum()}/{total}")
    return df, fps


# ──────────────────────────────────────────
# Automatic phase classification (rule-based state machine)
# ──────────────────────────────────────────
def classify_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically classifies six phases from wrist speed, knee height, and
    ankle-distance patterns.

    Classification logic:
        1) Compute the throwing wrist's frame-to-frame speed (throw_wrist_speed)
        2) Take the frame where speed reaches its overall max as the "release center"
        3) Walk forward/backward from the release center following the speed
           pattern to set the six phase boundaries:
           - windup: video start ~ the point the lead knee rises highest
           - stride: knee peak ~ front foot plant (max ankle y + sharp speed drop)
           - cocking: foot plant ~ the point wrist speed starts climbing in earnest
           - acceleration: the rising-speed window ~ just before the release center
           - release: release center +/- a small window (around the speed peak)
           - follow_through: after release ~ end of video
    """
    df = df.copy()
    n = len(df)
    df["phase"] = "windup"

    # fill in missing frames (detection failures are interpolated fore/aft for classification stability)
    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    for c in coord_cols:
        df[c] = df[c].interpolate(limit_direction="both")

    # smoothing
    df["wrist_x_s"] = df["throw_wrist_x"].rolling(3, center=True, min_periods=1).mean()
    df["wrist_y_s"] = df["throw_wrist_y"].rolling(3, center=True, min_periods=1).mean()
    df["knee_y_s"]  = df["lead_knee_y"].rolling(3, center=True, min_periods=1).mean()
    df["ankle_y_s"] = df["front_ankle_y"].rolling(3, center=True, min_periods=1).mean()

    # wrist speed
    vx = df["wrist_x_s"].diff()
    vy = df["wrist_y_s"].diff()
    df["wrist_speed"] = np.sqrt(vx**2 + vy**2)
    df["wrist_speed_s"] = df["wrist_speed"].rolling(3, center=True, min_periods=1).mean()

    if df["wrist_speed_s"].isna().all():
        return df  # can't classify - return with everything left as windup

    # ── 1) release center: point of peak wrist speed ──
    # Speed alone can also catch arm-flailing motion during the windup.
    # A real pitch's release/acceleration phase has a physical signature:
    # the throwing hand is above shoulder height, so add that as a filter.
    # Screen y increases downward, so "wrist above shoulder" means wrist_y < shoulder_y.
    df["throw_shoulder_y_s"] = df["throw_shoulder_y"].rolling(3, center=True, min_periods=1).mean()

    arm_raised_mask = df["wrist_y_s"] < df["throw_shoulder_y_s"]

    # down-weight frames where throw_wrist_vis is low (less reliable)
    reliable_mask = df.get("throw_wrist_vis", pd.Series(1, index=df.index)) > 0.4

    # release candidates = reliable AND arm raised above the shoulder
    candidate_mask = reliable_mask & arm_raised_mask
    speed_for_peak = df["wrist_speed_s"].where(candidate_mask, 0)

    if speed_for_peak.max() == 0:
        # if the arm never goes above the shoulder (pitching motion wasn't
        # captured, or detection quality is bad), relax the condition and
        # retry using speed alone
        speed_for_peak = df["wrist_speed_s"].where(reliable_mask, 0)

    release_center = int(speed_for_peak.idxmax())

    # ── 2) knee peak (windup -> stride boundary) ──
    # the point of smallest knee y (highest on screen), searched only
    # before the release center
    pre_release = df.loc[:release_center]
    if pre_release["knee_y_s"].notna().any():
        knee_peak_idx = int(pre_release["knee_y_s"].idxmin())
    else:
        knee_peak_idx = 0

    # ── 3) foot-plant moment (stride -> cocking boundary) ──
    # Problem with the previous version: searching the whole knee_peak_idx
    # ~ release_center window for "the first point the ankle stops moving"
    # could also match a static ready stance (legs spread, standing still),
    # mistaking it for foot plant.
    #
    # A real foot plant happens in a short window right before release, so
    # narrow the search to "just before release_center," and additionally
    # check for the inflection point where the knee starts bending again
    # after landing, to find the true plant instant (the split second the
    # leg is most extended).
    search_start = max(knee_peak_idx, release_center - int((release_center - knee_peak_idx) * 0.9))
    after_knee = df.loc[search_start:release_center].copy()

    if len(after_knee) > 3:
        ankle_vel = after_knee["ankle_y_s"].diff().abs()
        knee_vel  = after_knee["knee_y_s"].diff()  # positive = knee descending again (bending), negative = still extending

        y_thr = after_knee["ankle_y_s"].quantile(0.6)
        vel_thr = ankle_vel.quantile(0.5)

        # candidates: ankle low enough (plant height) + ankle velocity low
        # (stopped) + knee starts bending again shortly after (weight
        # transfer beginning)
        candidates = after_knee[(after_knee["ankle_y_s"] >= y_thr) & (ankle_vel < vel_thr)]

        landing_idx = None
        for cand_idx in candidates.index:
            # check whether the knee starts bending (knee_vel turns
            # positive) within 5 frames after this candidate
            window_end = min(cand_idx + 5, after_knee.index[-1])
            future_knee_vel = knee_vel.loc[cand_idx:window_end]
            if (future_knee_vel > 0).any():
                landing_idx = int(cand_idx)
                break

        if landing_idx is None:
            # if no candidate satisfies the knee-flexion condition, use the
            # stopping candidate closest to release (most likely to be the
            # plant right before release)
            landing_idx = int(candidates.index[-1]) if not candidates.empty else search_start
    else:
        landing_idx = search_start

    # ── 4) acceleration start (cocking -> acceleration boundary) ──
    # guarantee a minimum number of frames as cocking right after foot
    # plant first (the cocking motion itself moves the wrist a little, so
    # a raw speed threshold alone could jump straight past it), then look
    # for the first point after that where speed is "consistently rising."
    min_cocking_frames = max(2, int((release_center - landing_idx) * 0.15))
    cocking_floor_idx = min(landing_idx + min_cocking_frames, release_center)

    after_floor = df.loc[cocking_floor_idx:release_center]
    speed_threshold = df["wrist_speed_s"].max() * 0.25

    accel_start_idx = cocking_floor_idx
    if len(after_floor) >= 3:
        speeds = after_floor["wrist_speed_s"].values
        idxs = after_floor.index.values
        # first point where speed clears the threshold AND doesn't drop on
        # the next frame (rising trend)
        for k in range(len(speeds) - 1):
            if speeds[k] >= speed_threshold and speeds[k + 1] >= speeds[k]:
                accel_start_idx = int(idxs[k])
                break
        else:
            # if no point satisfies the rising-trend condition, use the
            # first point that simply clears the threshold
            over = after_floor[after_floor["wrist_speed_s"] >= speed_threshold]
            accel_start_idx = int(over.index[0]) if not over.empty else cocking_floor_idx

    # ── 5) width of the release window ──
    # classify release_center +/- 2 frames as "release" (a short window
    # around the speed peak)
    release_window = 2
    release_start = max(0, release_center - release_window)
    release_end   = min(n - 1, release_center + release_window)

    # ── apply labels ──
    df.loc[0:knee_peak_idx, "phase"] = "windup"
    df.loc[knee_peak_idx:landing_idx, "phase"] = "stride"
    df.loc[landing_idx:accel_start_idx, "phase"] = "cocking"
    df.loc[accel_start_idx:release_start, "phase"] = "acceleration"
    df.loc[release_start:release_end, "phase"] = "release"
    df.loc[release_end:n, "phase"] = "follow_through"

    df.attrs["release_center"] = release_center
    df.attrs["knee_peak_idx"] = knee_peak_idx
    df.attrs["landing_idx"] = landing_idx
    df.attrs["accel_start_idx"] = accel_start_idx

    return df


# ──────────────────────────────────────────
# HSS calculation (release-phase frames only)
# ──────────────────────────────────────────
def measure_hss_at_release(df: pd.DataFrame):
    """
    Computes HSS only on frames classified as phase == 'release', and
    returns the max value within the normal range.

    Double safety net: classify_phases already gated release_center on
    "wrist above shoulder," but frames at the edge of the release_window
    (+/-2 frames) could fall outside that condition, so the same check is
    applied again here to keep only frames where the arm is genuinely raised.

    Returns: (hss_value, frame_idx) or (None, None)
    """
    release_df = df[df["phase"] == "release"].copy()
    if release_df.empty:
        return None, None

    hss_list = []
    for idx, row in release_df.iterrows():
        if pd.isna(row.get("l_hip_x")) or pd.isna(row.get("r_shoulder_x")):
            continue

        # re-verify the throwing hand is above the shoulder (screen y gets
        # smaller going up)
        wrist_y = row.get("throw_wrist_y")
        shoulder_y = row.get("throw_shoulder_y")
        if pd.notna(wrist_y) and pd.notna(shoulder_y) and wrist_y >= shoulder_y:
            continue  # arm below shoulder = not a throwing motion, exclude

        l_hip = np.array([row["l_hip_x"], row["l_hip_y"]])
        r_hip = np.array([row["r_hip_x"], row["r_hip_y"]])
        l_sh  = np.array([row["l_shoulder_x"], row["l_shoulder_y"]])
        r_sh  = np.array([row["r_shoulder_x"], row["r_shoulder_y"]])

        hip_angle = axis_angle(l_hip, r_hip)
        sh_angle  = axis_angle(l_sh, r_sh)
        hss = hip_angle - sh_angle
        if hss > 180:
            hss -= 360
        elif hss < -180:
            hss += 360

        hss_list.append((idx, hss))

    if not hss_list:
        return None, None

    hss_series = pd.Series({i: v for i, v in hss_list})
    valid = hss_series[(hss_series >= HSS_MIN) & (hss_series <= HSS_MAX)]

    if valid.empty:
        # even within the release window, if only outlier values remain, treat as a failed measurement
        return None, None

    best_idx = int(valid.idxmax())
    return float(valid.max()), int(df.loc[best_idx, "frame"])


# ──────────────────────────────────────────
# Stride calculation (at the stride -> cocking boundary, right at/around foot plant)
# ──────────────────────────────────────────
def measure_stride_at_landing(df: pd.DataFrame, player_height_m: float):
    """
    Restricts candidates to frames where "the throwing wrist is above the
    shoulder," then finds the point among those with the largest stride
    (front-to-back ankle distance).

    Why this changed: the previous version relied on a single landing_idx
    frame found by classify_phases, but a static ready stance (legs spread,
    standing still) could sometimes be mistaken for foot plant.
    "Hand raised above the shoulder" is strong physical evidence that the
    pitching motion is actually in progress (around cocking~release), so
    measuring stride only on frames meeting that condition rules out a
    static stance at the source.
    (Screen y decreases going up, so "wrist above shoulder" = wrist_y < shoulder_y)
    """
    if "throw_wrist_y" not in df.columns or "throw_shoulder_y" not in df.columns:
        return None, None

    arm_raised = df["throw_wrist_y"] < df["throw_shoulder_y"]
    candidates = df[arm_raised].copy()

    if candidates.empty:
        return None, None  # the hand never went above the shoulder

    shoulder_w_median = df["shoulder_w_px"].median()
    if pd.isna(shoulder_w_median) or shoulder_w_median <= 0:
        return None, None
    scale = 0.45 / shoulder_w_median

    best_stride_m = None
    best_idx = None

    for idx, row in candidates.iterrows():
        if pd.isna(row.get("front_ankle_x")) or pd.isna(row.get("back_ankle_x")):
            continue

        f_xy = np.array([row["front_ankle_x"], row["front_ankle_y"]])
        b_xy = np.array([row["back_ankle_x"], row["back_ankle_y"]])
        stride_px = dist(f_xy, b_xy)
        stride_m = stride_px * scale
        stride_ratio = stride_m / player_height_m

        # exclude out-of-range candidates
        if not (STRIDE_RATIO_MIN <= stride_ratio <= STRIDE_RATIO_MAX):
            continue

        if best_stride_m is None or stride_m > best_stride_m:
            best_stride_m = stride_m
            best_idx = idx

    if best_stride_m is None:
        return None, None

    return float(best_stride_m), int(df.loc[best_idx, "frame"])


# ──────────────────────────────────────────
# Height-based scale (more accurate than assuming a fixed shoulder width)
# ──────────────────────────────────────────
def estimate_height_based_scale(df: pd.DataFrame, player_height_m: float) -> float | None:
    """
    The previous version derived the pixel-to-meter scale from a fixed
    assumption of "shoulder width = 0.45m." Since body proportions vary
    from person to person, this can be inaccurate, so instead derive the
    scale from the player's actual entered height.

    Method: treat the pixel distance from the nose to the midpoint of the
    ankles as an approximate "head-to-toe height" in the video, and
    calibrate so that distance corresponds to the real player_height_m.
    Measured across many frames and reduced to a median to cut down noise.

    Returns: real-world meters per pixel (scale), or None if it can't be measured
    """
    required = ["nose_x", "nose_y", "l_ankle_x", "l_ankle_y", "r_ankle_x", "r_ankle_y"]
    if not all(c in df.columns for c in required):
        return None

    valid = df.dropna(subset=required)
    if valid.empty:
        return None

    nose = valid[["nose_x", "nose_y"]].values
    ankle_mid_x = (valid["l_ankle_x"] + valid["r_ankle_x"]) / 2
    ankle_mid_y = (valid["l_ankle_y"] + valid["r_ankle_y"]) / 2
    ankle_mid = np.stack([ankle_mid_x.values, ankle_mid_y.values], axis=1)

    px_heights = np.linalg.norm(nose - ankle_mid, axis=1)
    # smooth out outliers (frames where the person appears small in a
    # corner of frame, etc.) using the median
    px_height_median = np.median(px_heights)

    if px_height_median <= 0:
        return None

    # nose-to-ankle distance is slightly shorter than true crown-to-toe
    # height (the nose sits below the crown of the head), so apply a ~1.05x
    # correction to account for the extra height above the nose
    HEAD_TOP_CORRECTION = 1.05
    scale = (player_height_m / HEAD_TOP_CORRECTION) / px_height_median
    return float(scale)


# ──────────────────────────────────────────
# Release Extension (release distance relative to the rubber)
# ──────────────────────────────────────────
def measure_release_extension(df: pd.DataFrame, player_height_m: float,
                                player_wingspan_m: float, max_hss_frame: int = None):
    """
    Computes how far in front of the rubber (approximated by the back foot
    position at the start of the delivery) the ball is released. Same
    concept as MLB's "Extension" metric.

    Method:
        1) Convert pixels to meters using the height-based scale (more
           accurate than assuming a shoulder width)
        2) Among frames where "the wrist is above the shoulder" (candidate
           cocking~release window), take the point where the wrist reaches
           farthest forward (toward the target) as the release moment.
        3) Rubber reference point = the back-foot position at that moment
           (the back foot stays roughly fixed near the rubber throughout
           the delivery)
        4) Extension = horizontal distance from the rubber position to the
           release wrist position
        5) Arm-length correction via the wingspan/height ratio:
           someone whose wingspan exceeds their height (true for most
           people) tends to have a longer reach and thus a larger
           extension in the same posture. A wingspan/height ratio is
           blended lightly into the raw pixel measurement to fold arm
           length into the estimate.

    Returns: (extension_m, frame_idx) or (None, None)
    """
    if "throw_wrist_x" not in df.columns or "throw_shoulder_y" not in df.columns:
        return None, None

    scale = estimate_height_based_scale(df, player_height_m)
    if scale is None:
        # fall back to the shoulder-width assumption if the height-based scale can't be computed
        shoulder_w_median = df["shoulder_w_px"].median()
        if pd.isna(shoulder_w_median) or shoulder_w_median <= 0:
            return None, None
        scale = 0.45 / shoulder_w_median

    arm_raised = df["throw_wrist_y"] < df["throw_shoulder_y"]
    candidates = df[arm_raised].copy()
    if candidates.empty:
        return None, None

    # estimate the pitching direction: determine across the whole video
    # whether "back_ankle -> front_ankle" points toward increasing or
    # decreasing x
    valid_ankles = df.dropna(subset=["front_ankle_x", "back_ankle_x"])
    if valid_ankles.empty:
        return None, None
    direction_sign = np.sign(
        (valid_ankles["front_ankle_x"] - valid_ankles["back_ankle_x"]).median()
    )
    if direction_sign == 0:
        direction_sign = 1

    # the moment the wrist reaches farthest along the pitching direction = release moment
    candidates["wrist_forward"] = candidates["throw_wrist_x"] * direction_sign
    release_idx = candidates["wrist_forward"].idxmax()
    release_row = df.loc[release_idx]

    if pd.isna(release_row.get("back_ankle_x")) or pd.isna(release_row.get("throw_wrist_x")):
        return None, None

    # rubber reference point = the back-foot position at that moment
    rubber_x = release_row["back_ankle_x"]
    wrist_x = release_row["throw_wrist_x"]

    extension_px = abs(wrist_x - rubber_x)
    extension_m = extension_px * scale

    # arm-length correction via wingspan/height ratio
    wingspan_ratio = player_wingspan_m / player_height_m
    extension_m_corrected = extension_m * (0.5 + 0.5 * wingspan_ratio)
    # (weighting the correction factor between 0.5-1.0 keeps it gentle and avoids over-correcting)

    # sanity range check: extension rarely exceeds ~0.5-1.3x height
    if not (0.3 * player_height_m <= extension_m_corrected <= 1.3 * player_height_m):
        return None, None

    return float(extension_m_corrected), int(release_row["frame"])


# ──────────────────────────────────────────
# Visualization: phase timeline
# ──────────────────────────────────────────
def plot_phase_timeline(df: pd.DataFrame, save_path: str, title: str):
    """
    Draws the six phases as colored bars, labeling each with its exact
    start-end time and duration (seconds). When a segment is too narrow to
    fit the label (especially a momentary one like release), the label is
    pulled above the bar and pointed to with an arrow instead.
    """
    fig, ax = plt.subplots(figsize=(13, 3.6))

    t = df["time_sec"].values

    # find each phase's contiguous segments (start, end) — cuts a new
    # segment every time the phase value changes.
    # (assumes a phase doesn't appear in multiple disjoint segments within
    #  one video, but even if it does, every contiguous run is still drawn
    #  safely.)
    phase_series = df["phase"].values
    segments = []  # (phase_name, start_time, end_time)
    seg_start_idx = 0
    for i in range(1, len(phase_series) + 1):
        if i == len(phase_series) or phase_series[i] != phase_series[seg_start_idx]:
            segments.append((
                phase_series[seg_start_idx],
                t[seg_start_idx],
                t[i - 1],
            ))
            seg_start_idx = i

    # draw the bars
    for phase_name, t_start, t_end in segments:
        ax.axvspan(t_start, t_end, color=PHASE_COLORS_HEX.get(phase_name, "#999"),
                   alpha=0.85, ymin=0.1, ymax=0.9)

    # decide the minimum bar width that can fit an inline label, based on
    # the total timeline span (previously 0.05, raised to be more generous
    # since short windows like cocking/acceleration at 0.15-0.2s were
    # causing the text to overlap the neighboring segment)
    total_span = t[-1] - t[0] if len(t) > 1 else 1
    min_width_for_inline_label = total_span * 0.12

    label_y_inline = 0.5      # label height inside the bar
    outside_label_levels = [1.05, 1.65, 1.05]  # cycle outside-label height
                                                  # through 3 levels so back-
                                                  # to-back narrow segments don't overlap
    outside_label_count = 0

    for phase_name, t_start, t_end in segments:
        duration = t_end - t_start
        mid = (t_start + t_end) / 2
        # name / start-end (duration) — kept to 2 lines so it fits safely inside the box height
        label_text = f"{phase_name}\n{t_start:.2f}-{t_end:.2f}s ({duration:.2f}s)"

        if duration >= min_width_for_inline_label:
            # wide enough to fit the label inside the bar — center it there
            ax.text(mid, label_y_inline, label_text, ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold",
                    linespacing=1.6)
        else:
            # narrow segments (cocking, acceleration, release, etc.) get
            # their label pulled above the bar and connected with an
            # arrow. Since space isn't constrained out there, the outside
            # label uses 3 lines for readability.
            label_text_outside = f"{phase_name}\n{t_start:.2f}-{t_end:.2f}s\n({duration:.2f}s)"
            y_pos = outside_label_levels[outside_label_count % len(outside_label_levels)]
            outside_label_count += 1
            ax.annotate(
                label_text_outside,
                xy=(mid, 0.9), xytext=(mid, y_pos),
                ha="center", va="bottom", fontsize=7.5, color="#333",
                linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color="#888", lw=1),
            )

    ax.set_ylim(0, 2.6)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(title, fontsize=12)

    # keep a separate legend at the bottom for color-to-phase-name matching
    handles = [mpatches.Patch(color=PHASE_COLORS_HEX[p], label=p) for p in PHASES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=6, fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved phase timeline: {save_path}")


# ──────────────────────────────────────────
# Overlay images for validation
# ──────────────────────────────────────────
def save_overlay_image(video_path, model_path, target_frame, label, save_path, mode="hss",
                        num_poses=1, pitcher_select="smallest", throws_right=True):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  (validation image failed: couldn't read frame {target_frame})")
        return

    scale_factor = 3 if w > 1500 else 1
    proc = cv2.resize(frame, (w // scale_factor, h // scale_factor)) if scale_factor > 1 else frame
    ph, pw = proc.shape[:2]

    with create_landmarker(model_path, RunningMode.IMAGE, num_poses=num_poses) as lmkr:
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = lmkr.detect(mp_img)

    lms = select_pitcher_pose(result.pose_landmarks, mode=pitcher_select) if result.pose_landmarks else None
    if lms is None:
        print(f"  (validation image: no pose detected on frame {target_frame})")
        return

    def pt(idx):
        return (int(lms[idx].x * pw), int(lms[idx].y * ph))

    if mode == "stride":
        cv2.line(proc, pt(IDX["l_ankle"]), pt(IDX["r_ankle"]), (0, 165, 255), 4)
    elif mode == "extension":
        throw_wrist_idx = IDX["r_wrist"] if throws_right else IDX["l_wrist"]
        back_ankle_idx = IDX["r_ankle"] if throws_right else IDX["l_ankle"]
        cv2.line(proc, pt(back_ankle_idx), pt(throw_wrist_idx), (0, 220, 255), 4)
        cv2.circle(proc, pt(back_ankle_idx), 10, (0, 140, 255), 2)  # highlight the rubber position
    else:
        cv2.line(proc, pt(IDX["l_hip"]), pt(IDX["r_hip"]), (0, 165, 255), 3)
        cv2.line(proc, pt(IDX["l_shoulder"]), pt(IDX["r_shoulder"]), (180, 60, 220), 3)

    for idx in IDX.values():
        if lms[idx].visibility > 0.3:
            cv2.circle(proc, pt(idx), 6, (0, 200, 120), -1)

    cv2.putText(proc, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 150), 2)
    cv2.putText(proc, f"Frame: {target_frame}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    cv2.imwrite(str(save_path), proc)
    print(f"  Saved validation image: {save_path}")


# ──────────────────────────────────────────
# Automatic video discovery & selection
# ──────────────────────────────────────────
def select_video_interactively(search_dir: str, extensions: list) -> str:
    """
    Finds video files in search_dir and shows them as a numbered list for
    the user to pick from. This means never having to edit a filename in
    the code when you shoot a new video — just drop it in this folder.

    To use a file that isn't in the list, choose option 0 to type a path directly.
    """
    search_path = Path(search_dir)
    videos = sorted([
        p for p in search_path.iterdir()
        if p.is_file() and p.suffix in extensions
    ])

    if not videos:
        print(f"No video files found in '{search_dir}'.")
        manual = input("Type the filename (or path) of the video to analyze: ").strip()
        return manual

    print("\nSelect a video to analyze:")
    for i, v in enumerate(videos, start=1):
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {v.name}  ({size_mb:.1f}MB)")
    print(f"  [0] A different video not in this list — type the filename directly")

    while True:
        choice = input(f"Enter a number (1-{len(videos)}, 0): ").strip()
        if choice == "0":
            manual = input("Type the filename (or path) of the video to analyze: ").strip()
            return manual
        if choice.isdigit() and 1 <= int(choice) <= len(videos):
            return str(videos[int(choice) - 1])
        print("Invalid input. Try again.")


def make_output_dir(video_path: str) -> Path:
    """
    Creates a results folder named after the video file.
    e.g. IMG_9225_2.mov -> results/IMG_9225_2/
    Running multiple videos won't overwrite each other's results — each
    video gets its own folder.
    """
    video_stem = Path(video_path).stem
    out_dir = Path("results") / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_PATH = select_video_interactively(VIDEO_SEARCH_DIR, VIDEO_EXTENSIONS)
    OUT_DIR = make_output_dir(VIDEO_PATH)
    print(f"\nSelected video: {VIDEO_PATH}")
    print(f"Saving results to: {OUT_DIR}/\n")

    print(f"[{VIDEO_PATH}] Extracting joints...")
    df, fps = extract_all_landmarks(
        VIDEO_PATH, MODEL_PATH, THROWS_RIGHT,
        num_poses=NUM_POSES, pitcher_select=PITCHER_SELECT_MODE
    )
    print("Classifying motion phases...")
    df = classify_phases(df)
    plot_phase_timeline(df, str(OUT_DIR / "phase_timeline.png"), "Pitching Motion Phases (Side View)")

    stride_m, stride_frame = measure_stride_at_landing(df, PLAYER_HEIGHT_M)
    extension_m, extension_frame = measure_release_extension(df, PLAYER_HEIGHT_M, PLAYER_WINGSPAN_M)
    max_hss, max_hss_frame = measure_hss_at_release(df)

    print("\n" + "="*40)
    print("  Pitching Mechanics Analysis Results (measured at release)")
    print("="*40)

    if stride_m is None:
        print("Stride length: measurement failed (no foot-plant moment or no in-range value found)")
    else:
        print(f"Stride length: {stride_m:.2f}m  (at the foot-plant frame)")
        save_overlay_image(VIDEO_PATH, MODEL_PATH, stride_frame,
                            f"Stride: {stride_m:.2f}m", str(OUT_DIR / "stride_check_frame.png"), mode="stride",
                            num_poses=NUM_POSES, pitcher_select=PITCHER_SELECT_MODE,
                            throws_right=THROWS_RIGHT)

    if extension_m is None:
        print("Release extension: measurement failed (no in-range value found within the release window)")
    else:
        print(f"Release extension: {extension_m:.2f}m  (release distance relative to the rubber, height/wingspan-adjusted)")
        save_overlay_image(VIDEO_PATH, MODEL_PATH, extension_frame,
                            f"Extension: {extension_m:.2f}m", str(OUT_DIR / "extension_check_frame.png"), mode="extension",
                            num_poses=NUM_POSES, pitcher_select=PITCHER_SELECT_MODE,
                            throws_right=THROWS_RIGHT)

    if max_hss is None:
        print("Max hip-shoulder separation: measurement failed (no in-range value found within the release window)")
    else:
        print(f"Max hip-shoulder separation: {max_hss:.1f} deg  (at a release-phase frame)")
        save_overlay_image(VIDEO_PATH, MODEL_PATH, max_hss_frame,
                            f"HSS: {max_hss:.1f} deg (release)", str(OUT_DIR / "hss_check_frame.png"), mode="hss",
                            num_poses=NUM_POSES, pitcher_select=PITCHER_SELECT_MODE,
                            throws_right=THROWS_RIGHT)

    print("="*40)
    print(f"\nResults folder: {OUT_DIR}/")
    print("Check the phase timeline image (phase_timeline.png) to see")
    print("where the release/stride windows landed.")