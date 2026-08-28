# Low-Cost 3D Pitching Biomechanics Analysis Using Smartphones

**Braylon** · St. Mark's School, Class of 2027
braylonkim@stmarksschool.org

---

<img width="480" height="854" alt="all_pitches_overlay" src="https://github.com/user-attachments/assets/7ad1d258-9510-407b-84b0-3093bd4be12c" />

## Overview

As a varsity baseball pitcher stuck in a velocity plateau, I wanted objective biomechanical feedback without access to expensive motion-capture systems (Rapsodo, TrackMan, etc.), which typically cost thousands of dollars and aren't practical for a high school athlete to use on a daily basis. Coaches can watch a bullpen session and offer cues, but they can't tell you that your hip-shoulder separation was 15° instead of 30°, or that your stride shrank by 20cm between your first and fifteenth pitch of the day. That kind of precise, repeatable measurement usually requires a lab.

So I built one — out of two phones.

This project is a computer vision pipeline that measures three key indicators of pitching efficiency directly from ordinary smartphone video:

- **Hip-Shoulder Separation (HSS)** — how much the hips rotate open before the shoulders catch up, in 3D
- **Stride Length** — the distance between the back foot (rubber) and the front foot at landing
- **Release Extension** — how far in front of the rubber the ball is released

The system uses two synchronized phone cameras, Google's MediaPipe pose estimation model, and classical stereo computer vision (Essential Matrix recovery + triangulation) to reconstruct the pitcher's joints in real 3D space — not just as a flat, foreshortened 2D projection — from footage anyone can capture with equipment they already own.

---
## Motivation & Hypotheses

Elite pitchers generate higher velocity largely through more efficient use of the kinetic chain — energy transferred sequentially from the ground up through the hips, trunk, shoulder, elbow, and finally the hand, each segment moving faster than the one before it. Two of the clearest, most measurable signatures of this sequencing are how much the hips "lead" the shoulders during rotation, and how far the body travels toward the plate before the ball is released. I hypothesized that, within my own pitches:

1. Larger hip-shoulder separation near front-foot-plant correlates with higher velocity
2. Greater stride length and release extension (relative to height) also correlate with higher velocity

Velocity correlation analysis is still in progress — I'm collecting radar-gun-matched pitch data across multiple sessions to build a dataset large enough to say something statistically meaningful, rather than drawing a trend line through three data points.

---

## Experimental Setup

- **Cameras:** Two iPhones (slow-motion or regular mode, 30–240 fps), one positioned roughly side-on to the pitcher and one behind/near the catcher, angled apart by as wide a baseline as the field allows
- **Software:** Python, OpenCV, MediaPipe PoseLandmarker, Streamlit (for the interactive dashboard)
- **Calibration:** No checkerboard or reference object needed — the system self-calibrates the stereo pair directly from the pitcher's own tracked motion, then scales the reconstruction using the pitcher's known height (nose-to-ankle) and a standard height-to-wingspan ratio
- **Ground truth:** Radar gun for velocity, used to validate whether the measured mechanics actually predict pitch speed

The pipeline syncs the two videos (cross-correlating the audio tracks, then letting the user nudge cam1 and cam2 independently frame-by-frame to confirm), recovers the relative camera pose via the Essential Matrix (`cv2.USAC_MAGSAC`), triangulates 3D joint positions for every matched frame pair, and then computes the three metrics above at the biomechanically correct instant in the delivery.

![Camera Setup](https://github.com/user-attachments/assets/52544ad7-465c-48d4-99e4-dddd8fc180db)

*Two-camera side-view setup used for 3D triangulation*

---

## Pipeline Architecture

1. **Sync** — Audio cross-correlation gives a starting offset estimate; the dashboard then lets you visually confirm (or correct) it by scrubbing both camera feeds independently until the same instant lines up in both.
2. **Pose extraction** — MediaPipe's PoseLandmarker (heavy model) runs on both videos independently, producing 33 3D landmarks per frame with per-landmark visibility/confidence scores.
3. **Stereo calibration** — Matched frame pairs feed into Essential Matrix estimation, decomposed into relative rotation and translation between the two cameras. A held-out consistency check (comparing calibration results from alternating/interleaved frame subsets) flags sessions where the calibration isn't trustworthy.
4. **Triangulation** — Every matched pair of 2D detections is triangulated into a single 3D point, giving a full 3D skeleton over time, scaled to real-world units via the pitcher's known height.
5. **Event detection** — The pipeline finds each pitch by looking for a sustained window where the throwing wrist is above the shoulder, combined with a minimum peak angular velocity, so that a normal standing pose or an equipment adjustment doesn't get mistaken for a pitch.
6. **Metric extraction** — For each detected pitch, HSS is read at the true front-foot-plant instant (not an arbitrary smoothed window — more on why below), stride length is measured between the back and front ankle at that same plant frame, and extension is measured at the moment the throwing wrist reaches its farthest point toward the plate.
7. **Output** — Per-pitch annotated images (skeleton overlay + measurement), a time-series plot of HSS across the whole session, and a synchronized dual-camera overlay video, all generated automatically and served through a Streamlit dashboard.


<img width="1218" height="263" alt="Screenshot 2026-08-28 at 7 16 52 PM" src="https://github.com/user-attachments/assets/fadd75d1-b0c9-4a77-8933-ba4087bda4de" />


---



## Key Technical Challenges Solved

Building this pipeline surfaced far more edge cases than I expected going in. A few of the most instructive ones:

| Problem | Solution |
|---|---|
| Catcher misidentified as pitcher | Select the *smallest* person in the frame (farthest from a side-view camera) |
| Static stance counted as a pitch | Require throwing wrist above shoulder + minimum peak angular velocity |
| Single-camera angle severely under-estimated separation/extension | Switched from monocular estimation to full stereo 3D triangulation |
| Peak HSS jumped around due to single-frame tracking noise | Anchor HSS to the precise, independently-detected front-foot-plant frame instead of blindly taking whichever frame in a window has the largest angle |
| A brief, real gap in tracking data got wrongly merged with several seconds of unrelated post-pitch noise into one "event" | Tightened the gap-merging tolerance so a real ~0.15s release window doesn't get diluted by 2+ seconds of an unrelated later moment |
| HSS sign flipped between otherwise-identical runs with a slightly different sync offset | Traced this to the self-calibration's coordinate system chirality being arbitrary and offset-dependent; fixed by anchoring the sign to the *un-triangulated* 2D pixel geometry from one camera, which has no such ambiguity |
| Camera occlusion made the throwing arm's tracked position confidently wrong (high reported confidence, visibly incorrect position) right at release | Cross-checked the release-arm position against the *other* camera's independent tracking of the same instant, and used a visible-ball detection pass to directly confirm the true release frame against raw pixels |
| Stride measured at the release frame was silently too short | Realized the back (push-off) foot drags forward during follow-through — stride has to be measured at front-foot-plant specifically, which is a different, earlier frame than release |
| Browser-unplayable overlay videos | Re-encode with ffmpeg (H.264, `yuv420p`) after OpenCV's native `mp4v` output |
| Multi-pitch overlay videos looked unsynchronized | Switched the alignment anchor from a fixed leg-height threshold (which different deliveries cross at different points in the motion) to each pitch's own local peak leg-lift height, a more biomechanically consistent reference point across pitches |

The common thread across almost all of these: a computer vision pipeline built on a single heuristic (a wrist height, a fixed frame window, one camera's raw output) will *look* like it works right up until it meets a case that heuristic wasn't written for. Nearly every fix above came from directly inspecting raw video frames against what the numbers claimed, rather than trusting an intermediate value because it came out of a "3D" calculation.

---

## Current Status & Next Steps

The system currently outputs per-pitch measurements, time-series plots of hip-shoulder separation, per-pitch annotated stride/extension/HSS images, and synchronized dual-camera overlay videos, all through a Streamlit dashboard that runs the whole pipeline end-to-end from two uploaded video files.

**Still to do:**
- Collect a larger radar-gun-matched dataset across multiple sessions and run the actual velocity correlation analysis the hypotheses above are waiting on
- Improve calibration stability for shorter clips (a single pitch's worth of footage doesn't give the stereo calibration enough motion to converge as reliably as a full multi-pitch bullpen does)
- Extend release-point detection to automatically cross-check against the second camera whenever the primary camera's tracking confidence is low, rather than requiring a manual review
- General robustness pass so a first-time user pointing two phones at a bullpen session gets credible results without needing to manually inspect intermediate frames the way I did throughout development


