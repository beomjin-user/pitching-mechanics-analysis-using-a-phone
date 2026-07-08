"""
test_synthetic.py — Validates geometry.py against synthetic ground truth.

There's no checkerboard here to validate calibration accuracy against, so
this is the next best thing: generate a fake 3D scene + two fake cameras
with an EXACTLY KNOWN relative pose, project to 2D, add MediaPipe-realistic
pixel noise, run the same estimation as run_calibration.py, and measure how
far the *estimated* relative pose is from the *known* one.

Run this whenever you touch geometry.py's estimation method — it's the
sanity check that catches regressions (e.g. it's what shows cv2.RANSAC is
much worse than cv2.USAC_MAGSAC at this noise level for this kind of
correspondence set).
"""

import numpy as np
import cv2

IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1920
FOCAL_PX = 1300.0  # rough phone-camera-equivalent focal length in pixels


def random_rotation(max_deg=25):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def make_synthetic_scene(n_points=200, depth_range=(2000, 5000)):
    """3D points roughly in front of both cameras, in mm — stands in for
    the cloud of body keypoints collected across many video frames."""
    pts = np.random.uniform(-800, 800, size=(n_points, 3))
    pts[:, 2] = np.random.uniform(depth_range[0], depth_range[1], size=n_points)
    return pts


def project(pts3d, R, T, focal_px, cx, cy):
    cam_pts = (R @ pts3d.T).T + T
    x = focal_px * cam_pts[:, 0] / cam_pts[:, 2] + cx
    y = focal_px * cam_pts[:, 1] / cam_pts[:, 2] + cy
    return np.stack([x, y], axis=1)


def normalize(points_px, focal_px, cx, cy):
    norm = np.empty_like(points_px)
    norm[:, 0] = (points_px[:, 0] - cx) / focal_px
    norm[:, 1] = (points_px[:, 1] - cy) / focal_px
    return norm


def angle_between_rotations(R1, R2):
    R_rel = R1.T @ R2
    cos_angle = np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def angle_between_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def run_trial(pixel_noise_std, method):
    pts3d = make_synthetic_scene()
    cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2

    R_true = random_rotation(max_deg=25)
    baseline_mm = np.random.uniform(600, 1000)
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)
    T_true = direction * baseline_mm

    px1 = project(pts3d, np.eye(3), np.zeros(3), FOCAL_PX, cx, cy)
    px2 = project(pts3d, R_true, T_true, FOCAL_PX, cx, cy)

    px1 = px1 + np.random.normal(0, pixel_noise_std, px1.shape)
    px2 = px2 + np.random.normal(0, pixel_noise_std, px2.shape)

    norm1 = normalize(px1, FOCAL_PX, cx, cy)
    norm2 = normalize(px2, FOCAL_PX, cx, cy)

    E, mask = cv2.findEssentialMat(
        norm1, norm2, focal=1.0, pp=(0.0, 0.0),
        method=method, prob=0.999, threshold=1.5 / FOCAL_PX,
    )
    if E is None or E.shape != (3, 3):
        return None

    _, R_est, T_est, _ = cv2.recoverPose(E, norm1, norm2, focal=1.0, pp=(0.0, 0.0), mask=mask)

    rot_err = angle_between_rotations(R_true, R_est)
    dir_err = angle_between_vectors(T_true, T_est.ravel())
    return rot_err, dir_err


def summarize(method_name, method, noise_levels, n_trials=30):
    print(f"\n=== {method_name} ===")
    for noise in noise_levels:
        rot_errs, dir_errs, failures = [], [], 0
        for _ in range(n_trials):
            result = run_trial(noise, method)
            if result is None:
                failures += 1
                continue
            rot_errs.append(result[0])
            dir_errs.append(result[1])
        if not rot_errs:
            print(f"  noise={noise}px: all {n_trials} trials failed (E estimation returned None)")
            continue
        note = f", {failures} failed" if failures else ""
        print(f"  noise={noise}px (n={len(rot_errs)}{note}): "
              f"rotation err {np.mean(rot_errs):5.2f} +/- {np.std(rot_errs):5.2f} deg, "
              f"direction err {np.mean(dir_errs):5.2f} +/- {np.std(dir_errs):5.2f} deg")


if __name__ == "__main__":
    np.random.seed(0)
    noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    summarize("cv2.RANSAC", cv2.RANSAC, noise_levels)
    summarize("cv2.USAC_MAGSAC", cv2.USAC_MAGSAC, noise_levels)
    print("\nMediaPipe pose landmarks on phone video are realistically in the "
          "~1-3px noise range at 1080p+. Compare the two blocks above at "
          "those noise levels before trusting either method.")
