"""
pitch_visualizer.py — Build scrubbable, DUAL-CAMERA overlay videos.

For each pitch detected by analyze_stereo_hss.py, renders a single clip that
shows BOTH camera views side by side (cam1 side-view | cam2 back-view),
each with:
  - dots on the major joints
  - shoulder axis + hip axis lines
and, shared across both views:
  - a live-updating 3D HSS angle readout, pulled from hss_timeseries.json
    (so the number tracks the real curve, not just a per-pitch average)

Saved as P{n}_overlay.mp4 inside the results folder — app.py's existing
"Overlay Video" section already looks for exactly this filename and plays
it with st.video(), which gives a native scrub bar for free.

IMPORTANT: OpenCV's VideoWriter (fourcc "mp4v") produces a container most
browsers can't decode — it'll show up as a 0:00 / blank player in Streamlit
even though the file exists. So this script renders with OpenCV into a
throwaway temp file, then re-encodes with ffmpeg to H.264 (libx264 +
yuv420p), which every browser can play. ffmpeg must be on PATH
(macOS: `brew install ffmpeg`).

Usage:
    python3 pitch_visualizer.py <results_dir> <cam1_video> <cam2_video>

Example:
    python3 pitch_visualizer.py results/IMG_2425_2_2026-07-05_16-10 \\
        IMG_2425_2.mov Untitled.mov

Requires that analyze_stereo_hss.py has already been run on this video pair
AND that its results folder contains matches.json (a new export — re-run
analyze_stereo_hss.py once on this pair if your results folder predates it;
pitches.json / hss_timeseries.json must exist too).
"""
import sys
import os
import json
import shutil
import argparse
import subprocess
import cv2
import numpy as np
from pathlib import Path

# ─── must match analyze_stereo_hss.py's LM indices ──────────────────────────
LM = {
    "NOSE": 0,
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW":    13, "R_ELBOW":    14,
    "L_WRIST":    15, "R_WRIST":    16,
    "L_HIP":      23, "R_HIP":      24,
    "L_KNEE":     25, "R_KNEE":     26,
    "L_ANKLE":    27, "R_ANKLE":    28,
}
MAJOR_JOINTS = [
    "L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_WRIST", "R_WRIST",
    "L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE",
]

# ─── colours (BGR), matching analyze_stereo_hss.py ──────────────────────────
COL_DOT      = (0, 220, 80)
COL_SHOULDER = (0, 80, 220)
COL_HIP      = (0, 165, 255)
COL_TEXT_BG  = (20, 20, 20)
COL_TEXT     = (255, 255, 255)
DOT_R, LN = 8, 3

VIS_THR  = 0.4
PAD_SEC  = 0.6    # extra seconds shown before/after the pitch segment
TARGET_H = 720    # both views resized to this height before combining


def load_json(path):
    with open(path) as f:
        return json.load(f)


def nearest_angle(times, angles, t):
    """Nearest-neighbour lookup of the HSS angle at time t."""
    idx = int(np.argmin(np.abs(np.asarray(times) - t)))
    return float(angles[idx])


# NOTE: ASCII only in any string drawn with cv2.putText — OpenCV's built-in
# Hershey fonts can't render em dashes or the "deg" symbol and will print
# "???" instead (this bit us once already in analyze_stereo_hss.py).
def label(frame, text, pos, fscale=0.9, thick=2):
    x, y = max(0, pos[0]), max(24, pos[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 4), COL_TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fscale, COL_TEXT, thick)


def draw_pose(frame, landmarks):
    pts = {}
    for name in MAJOR_JOINTS + ["NOSE"]:
        lm = landmarks[LM[name]]
        if lm["visibility"] > VIS_THR:
            p = (int(lm["x"]), int(lm["y"]))
            pts[name] = p
            cv2.circle(frame, p, DOT_R, COL_DOT, -1)
    if "L_SHOULDER" in pts and "R_SHOULDER" in pts:
        cv2.line(frame, pts["L_SHOULDER"], pts["R_SHOULDER"], COL_SHOULDER, LN)
    if "L_HIP" in pts and "R_HIP" in pts:
        cv2.line(frame, pts["L_HIP"], pts["R_HIP"], COL_HIP, LN)


def resize_to_height(frame, target_h):
    h, w = frame.shape[:2]
    scale = target_h / h
    return cv2.resize(frame, (int(round(w * scale)), target_h))


def transcode_for_browser(temp_path, final_path):
    """Re-encode with libx264 so browsers (and Streamlit's st.video) can
    play it. See module docstring for why this step exists."""
    if shutil.which("ffmpeg") is None:
        print("  WARNING: ffmpeg not found on PATH - keeping the raw OpenCV "
              "file, but it likely won't play in the browser. Install "
              "ffmpeg (macOS: `brew install ffmpeg`) and re-run this script.")
        os.replace(temp_path, final_path)
        return

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(temp_path),
         "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(final_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg transcode failed (exit {result.returncode}):")
        if result.stderr:
            print("   ", result.stderr.decode(errors="ignore").strip().splitlines()[-1])
        os.replace(temp_path, final_path)
    else:
        os.remove(temp_path)


def build_dual_overlay(cam1_video, cam2_video, seg_matches, pitch,
                        times, angles, out_path):
    if not seg_matches:
        print(f"  P{pitch['n']}: no matched frames in this window, skipping")
        return

    # effective output fps from the actual spacing of matched frames, so
    # playback speed is correct regardless of each camera's native fps
    t1s = [m["t1"] for m in seg_matches]
    dt = float(np.median(np.diff(t1s))) if len(t1s) > 1 else (1 / 30)
    out_fps = 1.0 / dt if dt > 0 else 30.0

    cap1 = cv2.VideoCapture(cam1_video)
    cap2 = cv2.VideoCapture(cam2_video)
    fps1 = cap1.get(cv2.CAP_PROP_FPS) or out_fps
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or out_fps

    temp_path = out_path.with_name(f"_raw_{out_path.name}")
    writer = None

    for m in seg_matches:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, int(round(m["t1"] * fps1)))
        ok1, fr1 = cap1.read()
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(round(m["t2"] * fps2)))
        ok2, fr2 = cap2.read()
        if not ok1 or not ok2:
            continue

        draw_pose(fr1, m["lm1"])
        draw_pose(fr2, m["lm2"])
        fr1 = resize_to_height(fr1, TARGET_H)
        fr2 = resize_to_height(fr2, TARGET_H)
        combined = cv2.hconcat([fr1, fr2])

        angle = nearest_angle(times, angles, m["t1"])
        label(combined, f"Pitch {pitch['n']}  HSS: {angle:+.1f} deg", (16, 46), fscale=1.0)
        label(combined, f"Time: {m['t1']:.2f} s", (16, 88), fscale=0.8)
        label(combined, "Side view (cam1)", (16, TARGET_H - 20), fscale=0.7)
        label(combined, "Back view (cam2)", (fr1.shape[1] + 16, TARGET_H - 20), fscale=0.7)

        if writer is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_path), fourcc, out_fps, (w, h))
        writer.write(combined)

    cap1.release()
    cap2.release()
    if writer is None:
        print(f"  P{pitch['n']}: no frames could be read from either video, skipping")
        return
    writer.release()

    if not temp_path.exists() or temp_path.stat().st_size < 1024:
        print(f"  P{pitch['n']}: OpenCV wrote an empty/tiny file, skipping "
              f"(check that both videos actually cover this time range)")
        return

    transcode_for_browser(temp_path, out_path)
    print(f"  P{pitch['n']}_overlay.mp4 saved ({len(seg_matches)} frames, dual-view)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="results/<run> folder from analyze_stereo_hss.py")
    ap.add_argument("cam1_video", help="side-view video")
    ap.add_argument("cam2_video", help="back-view video")
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    missing = [f for f in ("pitches.json", "matches.json", "hss_timeseries.json")
               if not (out_dir / f).exists()]
    if missing:
        print(f"ERROR: missing {missing} in {out_dir}.")
        print("Re-run analyze_stereo_hss.py on this video pair first - "
              "matches.json is a new export, so an older results folder "
              "won't have it yet.")
        sys.exit(1)

    pitches = load_json(out_dir / "pitches.json")
    matches = load_json(out_dir / "matches.json")
    hss_ts  = load_json(out_dir / "hss_timeseries.json")
    times, angles = hss_ts["times"], hss_ts["angles_deg"]

    print(f"Rendering {len(pitches)} dual-view overlay video(s) to {out_dir}/ ...")
    for pitch in pitches:
        lo_t, hi_t = pitch["start_sec"] - PAD_SEC, pitch["end_sec"] + PAD_SEC
        seg_matches = [m for m in matches if lo_t <= m["t1"] <= hi_t]
        out_path = out_dir / f"P{pitch['n']}_overlay.mp4"
        build_dual_overlay(args.cam1_video, args.cam2_video, seg_matches,
                            pitch, times, angles, out_path)

    print("Done.")


if __name__ == "__main__":
    main()