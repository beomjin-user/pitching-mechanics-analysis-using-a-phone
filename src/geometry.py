"""
geometry.py — Core stereo self-calibration math (no checkerboard).

Pipeline:
  1. Take 2D MediaPipe keypoint correspondences from two unsynchronized,
     uncalibrated cameras.
  2. Normalize pixel coordinates into approximate camera-normalized
     coordinates using each phone's known horizontal FOV (this trades
     "calibrate everything from a checkerboard" for "use the phone's
     published optics + estimate only what a checkerboard would otherwise
     give us: the relative pose between the two cameras").
  3. Estimate the Essential Matrix with cv2.USAC_MAGSAC (see
     test_synthetic.py for why not plain cv2.RANSAC).
  4. Recover relative rotation R and translation direction T (unit norm).
  5. Triangulate all correspondence points -> 3D points, up to scale.
  6. Resolve the unknown scale using a known person height at a frame
     where they're standing upright.

LIMITS OF THIS APPROACH (read before trusting the numbers):
  - Step 2 uses a *known* FOV from phone spec sheets, not a checkerboard.
    If FOV_DEGREES_CAM1 / FOV_DEGREES_CAM2 below are wrong, every
    downstream number is wrong by roughly the same proportion.
  - MediaPipe keypoints are noisy — nowhere near the sub-pixel accuracy a
    checkerboard corner detector gets. test_synthetic.py shows this noise
    alone can swing a single estimate several degrees trial to trial.
    There is no per-run accuracy guarantee here, only a statistical one —
    that's why run_calibration.py does a split-half consistency check
    instead of trusting one estimate blindly.
"""

import subprocess
import json
import numpy as np
import cv2


# ---- Phone camera specs (EDIT to match your actual phones) ----
# Horizontal field of view of the MAIN (wide) camera, in degrees.
# Look up: "<your phone model> main camera horizontal field of view"
# NOTE: if you call estimate_fov_from_video() below, this fallback
# value is only used if EXIF extraction fails.
FOV_DEGREES_CAM1 = 73.0   # side-view (3rd-base line) camera
FOV_DEGREES_CAM2 = 73.0   # back-view (behind catcher) camera


# ── iPhone model → main-camera horizontal FOV lookup table ──────────────────
# Sources: Apple spec sheets + DXOMark measurements.
# "wide" = the standard/main lens (not ultra-wide, not telephoto).
# FOV values are for the MAIN lens at 1x zoom, landscape orientation.
_IPHONE_FOV_TABLE = {
    # iPhone 15 family
    "iphone 15 pro max": 69.7, "iphone 15 pro": 70.0,
    "iphone 15 plus":    72.6, "iphone 15":     72.6,
    # iPhone 14 family
    "iphone 14 pro max": 69.7, "iphone 14 pro": 69.7,
    "iphone 14 plus":    73.0, "iphone 14":     73.0,
    # iPhone 13 family
    "iphone 13 pro max": 69.7, "iphone 13 pro": 69.7,
    "iphone 13 mini":    72.6, "iphone 13":     72.6,
    # iPhone 12 family
    "iphone 12 pro max": 65.0, "iphone 12 pro": 65.0,
    "iphone 12 mini":    72.6, "iphone 12":     65.0,
    # iPhone 11 family
    "iphone 11 pro max": 65.0, "iphone 11 pro": 65.0,
    "iphone 11":         73.0,
    # older
    "iphone xs max": 65.0, "iphone xs": 65.0, "iphone xr": 73.0,
    "iphone x":      65.0, "iphone 8 plus": 73.0, "iphone 8": 73.0,
}

# ── Zoom-level correction ────────────────────────────────────────────────────
# When the user zooms in, the effective FOV shrinks proportionally.
# FOV_eff = 2 * arctan(tan(FOV_native/2) / zoom_factor)
def _apply_zoom(fov_native_deg, zoom_factor):
    if zoom_factor <= 0 or zoom_factor == 1.0:
        return fov_native_deg
    fov_half_rad = np.deg2rad(fov_native_deg / 2.0)
    fov_eff_rad  = 2 * np.arctan(np.tan(fov_half_rad) / zoom_factor)
    return float(np.rad2deg(fov_eff_rad))


def estimate_fov_from_video(video_path, fallback_fov=73.0):
    """
    Try to estimate the horizontal FOV of the camera that recorded
    `video_path`, using EXIF / QuickTime metadata extracted by ffprobe.

    Strategy (in order of reliability):
      1. Read focal_length and sensor width from EXIF → compute FOV directly.
      2. Match phone model string against _IPHONE_FOV_TABLE.
      3. Read zoom_factor tag (some iPhones write this) and apply correction.
      4. Fall back to `fallback_fov`.

    Returns (fov_deg: float, source: str) where `source` is a short string
    describing which strategy succeeded, for logging.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(video_path)],
            capture_output=True, timeout=15,
        )
        meta = json.loads(result.stdout)
    except Exception:
        return fallback_fov, "fallback (ffprobe failed)"

    # flatten all tags from all streams + format into one dict
    tags = {}
    for stream in meta.get("streams", []):
        tags.update({k.lower(): v for k, v in stream.get("tags", {}).items()})
    tags.update({k.lower(): v for k, v in meta.get("format", {}).get("tags", {}).items()})

    # ── strategy 1: focal length + sensor width ──────────────────────────────
    fl_mm   = None
    sw_mm   = None
    zoom    = 1.0

    for key in ("com.apple.quicktime.camera.focal_length",
                "focal_length", "exif:focallength"):
        if key in tags:
            try: fl_mm = float(str(tags[key]).split("/")[0]) / \
                          (float(str(tags[key]).split("/")[1]) if "/" in str(tags[key]) else 1.0)
            except Exception: pass
            break

    for key in ("com.apple.quicktime.camera.sensor_width_mm",
                "sensor_width", "exif:focallengthin35mmfilm"):
        if key in tags:
            try:
                val = str(tags[key])
                if "/" in val:
                    sw_mm = float(val.split("/")[0]) / float(val.split("/")[1])
                else:
                    sw_mm = float(val)
            except Exception:
                pass
            break

    # zoom factor
    for key in ("com.apple.quicktime.camera.zoom_factor",
                "zoom_factor", "zoom"):
        if key in tags:
            try: zoom = float(tags[key]); break
            except Exception: pass

    if fl_mm and sw_mm and fl_mm > 0 and sw_mm > 0:
        fov_native = float(np.rad2deg(2 * np.arctan(sw_mm / (2 * fl_mm))))
        fov_eff    = _apply_zoom(fov_native, zoom)
        return fov_eff, f"EXIF focal_length={fl_mm:.1f}mm sensor={sw_mm:.1f}mm zoom={zoom:.2f}x"

    # ── strategy 2: phone model table ────────────────────────────────────────
    model_str = tags.get("com.apple.quicktime.model", tags.get("model", "")).lower()
    for model_key, fov_native in _IPHONE_FOV_TABLE.items():
        if model_key in model_str:
            fov_eff = _apply_zoom(fov_native, zoom)
            src = f"model table ({model_key}, zoom={zoom:.2f}x)"
            return fov_eff, src

    # ── strategy 3: 35mm equivalent focal length only ────────────────────────
    for key in ("exif:focallengthin35mmfilm", "focallengthin35mmfilm"):
        if key in tags:
            try:
                fl35 = float(tags[key])
                # 35mm full-frame sensor width = 36mm
                fov_native = float(np.rad2deg(2 * np.arctan(36.0 / (2 * fl35))))
                fov_eff    = _apply_zoom(fov_native, zoom)
                return fov_eff, f"35mm-equiv focal={fl35:.0f}mm zoom={zoom:.2f}x"
            except Exception:
                pass

    return fallback_fov, f"fallback ({fallback_fov}°, no usable EXIF)"


def focal_pixels_from_fov(image_width_px, fov_degrees):
    """Convert a known horizontal FOV to an equivalent focal length in pixels."""
    fov_rad = np.deg2rad(fov_degrees)
    return (image_width_px / 2.0) / np.tan(fov_rad / 2.0)


def normalize_points(points_px, image_width, image_height, fov_degrees):
    """
    Convert pixel coordinates (N,2) to normalized camera coordinates
    (as if focal length = 1, principal point = (0,0)), using a known FOV
    instead of a checkerboard-derived intrinsic matrix.
    """
    points_px = np.asarray(points_px, dtype=np.float64)
    fx = focal_pixels_from_fov(image_width, fov_degrees)
    cx, cy = image_width / 2.0, image_height / 2.0
    norm = np.empty_like(points_px)
    norm[:, 0] = (points_px[:, 0] - cx) / fx
    norm[:, 1] = (points_px[:, 1] - cy) / fx
    return norm


def estimate_essential_matrix(norm1, norm2, confidence=0.999, norm_threshold=1.5 / 1400):
    """
    Estimate the Essential Matrix between two normalized point sets.
    Uses cv2.USAC_MAGSAC, NOT cv2.RANSAC.
    """
    E, mask = cv2.findEssentialMat(
        norm1, norm2, focal=1.0, pp=(0.0, 0.0),
        method=cv2.USAC_MAGSAC, prob=confidence, threshold=norm_threshold,
    )
    return E, mask


def recover_pose(E, norm1, norm2, mask=None):
    """Decompose E into rotation R and translation direction T (unit norm)."""
    _, R, T, mask_pose = cv2.recoverPose(
        E, norm1, norm2, focal=1.0, pp=(0.0, 0.0), mask=mask)
    return R, T, mask_pose


def triangulate_points(R, T, norm1, norm2):
    """
    Triangulate matched normalized points into 3D, up to unknown scale.
    """
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, np.asarray(T).reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, norm1.T, norm2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d


def resolve_scale(pts3d, idx_top, idx_bottom, known_distance_mm):
    """
    Resolve the unknown scale factor using a known real-world distance.
    """
    triangulated_distance = np.linalg.norm(pts3d[idx_top] - pts3d[idx_bottom])
    if triangulated_distance < 1e-9:
        raise ValueError("Triangulated reference distance is ~0 — bad frame or bad points")
    scale_factor = known_distance_mm / triangulated_distance
    return scale_factor, pts3d * scale_factor


def hip_shoulder_separation_angle(pts3d_mm, landmark_indices):
    """
    Compute the 3D hip-shoulder separation angle at a single frame.
    HSS is projected onto the horizontal plane (about the vertical axis).
    """
    ls = pts3d_mm[landmark_indices['L_SHOULDER']]
    rs = pts3d_mm[landmark_indices['R_SHOULDER']]
    lh = pts3d_mm[landmark_indices['L_HIP']]
    rh = pts3d_mm[landmark_indices['R_HIP']]

    shoulder_vec = rs - ls
    hip_vec = rh - lh

    def horizontal(v):
        return np.array([v[0], v[2]])

    sh2 = horizontal(shoulder_vec)
    hp2 = horizontal(hip_vec)
    sh2 = sh2 / np.linalg.norm(sh2)
    hp2 = hp2 / np.linalg.norm(hp2)

    cos_angle = np.clip(np.dot(sh2, hp2), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))

    cross_z = sh2[0] * hp2[1] - sh2[1] * hp2[0]
    return angle_deg if cross_z >= 0 else -angle_deg
