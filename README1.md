
# Smartphone-Based Pitching Biomechanics Analysis Using Computer Vision

> **Author:** [Braylon] ([St. Mark's School], Class of 2027)
> **Contact:** [braylonkim@stmarksschool.org] 

---

## 1. Executive Summary

This independent research project bridges athletic performance and data science. As a varsity baseball pitcher working through a velocity plateau, I built a computer vision pipeline that measures three kinematic indicators of pitching efficiency — **Hip-Shoulder Separation**, **Stride Length**, and **Release Extension** — from ordinary smartphone video, without relying on commercial motion-capture hardware.

A core part of this project was not just building the measurement pipeline, but **validating that its output is actually trustworthy** — several early versions produced numbers that looked plausible but were measuring the wrong thing entirely (see Section 5).

## 2. Problem Statement & Hypotheses

* **The Problem:** Professional-grade motion capture systems (e.g., Rapsodo, TrackMan, marker-based mocap suits) cost thousands of dollars and require facilities most high school athletes don't have access to. This makes biomechanical feedback — the kind that could help a pitcher diagnose *why* they're stuck at a certain velocity — largely inaccessible at the amateur level.
* **Hypothesis 1:** A larger Hip-Shoulder Separation angle near release reflects more efficient transfer of rotational energy from the lower body to the upper body (the kinetic chain), and should correlate positively with pitch velocity.
* **Hypothesis 2:** Greater Stride Length and Release Extension (relative to the pitcher's height) are associated with higher pitch velocity, by increasing the distance over which the body can accelerate the ball before release.

*(Hypotheses 1–2 are framed as the basis for ongoing data collection; the velocity-correlation analysis itself is still in progress — see Section 6.)*

## 3. Experimental Setup & Methodology

Video was collected using two smartphones camera positioned at a **90° side angle** relative to the pitching direction (3rd-base and 1st-base side). This angle was chosen deliberately after testing — see Section 5 for why front/rear angles distort the measurement.

* **Hardware Stack:**
  * Camera: iPhone, slow-motion mode (60–240fps depending on device)
  * Velocity Tracker: Radar gun (measures the dependent variable — output velocity in mph)
* **Software Stack:** Python 3.10, OpenCV, MediaPipe PoseLandmarker, Pandas, NumPy, Matplotlib
* **Calibration:** Pixel-to-metric scale is derived from the pitcher's known height (rather than an assumed shoulder-width constant), using the nose-to-ankle pixel distance as a reference. Release Extension is additionally adjusted using wingspan-to-height ratio.
<img width="2720" height="2040" alt="camera_setup_pitching_analysis_v4" src="https://github.com/user-attachments/assets/52544ad7-465c-48d4-99e4-dddd8fc180db" />


## 4. Kinematic Extraction Logic

**MediaPipe PoseLandmarker**, 33 landmarks/frame, from **two synchronized cameras** (side + back view). 3D triangulation before any measurement. Replaces earlier single-camera (2D monocular) version.

### A. Stereo Self-Calibration & Sync

```
audio sync → visual sync check → self-calibration (Essential Matrix) → 3D triangulation
```

- Audio cross-correlation for initial offset, then manual visual check: dashboard shows both cameras at ±3 candidate frame offsets around a known release moment, user picks the one where both views agree (ball in-hand vs. released should match).
- High audio-sync confidence ≠ frame-perfect. 1-2 frame error biases everything downstream.
- Relative camera R/T via Essential Matrix (`cv2.USAC_MAGSAC`).
- Validated with split-half consistency check - **interleaved** (odd/even), not chronological. Chronological split concentrates real motion in one half, starves the other, false-flags good calibrations as unstable.

### B. Hip-Shoulder Separation (3D, Δθ)

- Angle between 3D hip vector (23, 24) and 3D shoulder vector (11, 12), triangulated.
- Computed every frame, no phase classifier / no gating to a "release" window.
- Peak selected from a **smoothed** curve - raw argmax vulnerable to single-frame occlusion spikes.
- True 3D → immune to the single-camera foreshortening problem in 5.3. No optimal camera angle needed.

### C. Stride Length

Ankle-to-ankle (side camera), at the same frame as release (below) - not a separate foot-plant/knee-flexion detection.

### D. Release Extension

Back foot → throwing wrist, horizontal, at farthest-forward-wrist frame (arm raised above shoulder). Height-calibrated, wingspan-adjusted.

---

## 5. Errors, Trials, and Validation

1. **Wrong person as pitcher.** Unchanged: smallest/farthest bounding box selected.
2. **Static stance as pitch.** Wrist-above-shoulder alone still caught glove adjustments, catch-and-reset motions. Fix: require stride+extension present AND min hip-shoulder angular velocity. Fails either → flagged "rough estimate," not dropped.
3. **Camera angle distorting Δθ.** Solved structurally: 3D triangulation instead of single-camera-angle workaround. No longer angle-dependent.
4. **Video files lack photo-style focal-length EXIF.** FOV falls back to a default - same default across two different iPhone models → asymmetric calibration error. Still open.
5. **Chronological split-half looked unstable on good calibrations.** See 4A - fixed via interleaving.
6. **Peak-HSS frame noise-sensitive.** One occluded-landmark frame could spike and get picked as "the" peak. Fixed with smoothing pre-argmax.
7. **Overlay videos unplayable in browser.** OpenCV `mp4v` output isn't browser-decodable - showed as blank 0:00 player despite valid file. Fixed: re-encode via `ffmpeg` (H.264/yuv420p).

---

## 6. Current Data & Status

Ongoing collection. Per-session output (HSS/stride/extension per pitch, HSS-over-time plot, per-pitch overlay video w/ live joint tracking) generated and reviewed in Streamlit dashboard - no CSV log.

## 7. Limitations & Next Steps

- Stride/extension still 2D (side-camera image plane), unlike triangulated HSS.
- Pitch detection still heuristic (wrist height + rotation speed + stride/ext), not learned.
- FOV fallback still a single assumed default regardless of phone model - next: read model from metadata, look up true spec.
- Single-pitch basis, no cross-pitch averaging.
- No velocity data yet - correlation with measured velocity is next phase.