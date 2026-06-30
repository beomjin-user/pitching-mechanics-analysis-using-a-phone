
# ⚾ Biomechanical Pitching Mechanism Analysis Using Computer Vision

> **Author:** [Braylon] ([St. Mark's School], Class of 2027)
> **Topic:** Computer Vision (MediaPipe) based Pitching Kinematics Extraction & Velocity Correlation Study
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
* <img width="2720" height="1840" alt="camera_setup_pitching_analysis_v3" src="https://github.com/user-attachments/assets/8bf75397-e9f1-48cb-9b68-da5e7a040651" />

## 4. Kinematic Extraction Logic

The pipeline uses **MediaPipe PoseLandmarker** to extract 33 skeletal landmarks per frame, then runs each video through a **rule-based motion-phase classifier** before measuring anything.

### A. Six-Phase Motion Classification
Rather than searching the entire clip for a maximum value (which turned out to be unreliable — see Section 5), every frame is first labeled into one of six pitching phases based on wrist speed and joint trajectory:

```
windup → stride → cocking → acceleration → release → follow_through
```

Each metric below is only measured within the phase where it is physically meaningful.

### B. Hip-Shoulder Separation (Δθ)
Computed as the angular difference between the shoulder vector (landmarks 11, 12) and the hip vector (landmarks 23, 24), measured **only within frames classified as `release`**, and only when the throwing wrist is positioned above the throwing shoulder (a physical precondition for an actual release motion):

$$\theta_{\text{shoulder}} = \tan^{-1}\left(\frac{Y_{11} - Y_{12}}{X_{11} - X_{12}}\right), \quad \theta_{\text{hip}} = \tan^{-1}\left(\frac{Y_{23} - Y_{24}}{X_{23} - X_{24}}\right)$$
$$\Delta\theta = \theta_{\text{hip}} - \theta_{\text{shoulder}}$$

### C. Stride Length
Measured as the ankle-to-ankle distance at the frame where the lead foot lands, identified by a drop in vertical ankle velocity combined with a knee-flexion inflection point immediately after — not simply "the frame with the widest stance," which an earlier version mistakenly used.

### D. Release Extension
Measured as the horizontal distance between the back foot (an approximation of the pitching rubber) and the throwing wrist at the moment the wrist reaches its farthest point toward home plate, scaled using the height-based calibration and adjusted by the wingspan-to-height ratio.

## 5. Errors, Trials, and Validation

This section documents the debugging process, because most of the project's actual difficulty was here, not in writing the initial version of the code.

**1. Wrong person identified as the pitcher.** In footage filmed from behind the catcher, MediaPipe sometimes locked onto the catcher (closer to the camera, larger in frame) instead of the pitcher, producing a "hip-shoulder separation" of 80+ degrees — anatomically impossible.
→ *Fix:* When multiple people are detected, the smallest (farthest) person in frame is automatically selected as the pitcher.

**2. Static stance misidentified as the pitching motion.** Early versions searched the full clip for the frame with the largest stride distance or rotation angle, which sometimes picked a relaxed standing pose instead of an actual pitch.
→ *Fix:* Measurements are now restricted to frames where the throwing wrist is above the throwing shoulder — a condition only true during an actual cocking/release motion, not a resting stance.

**3. Camera angle distorted the separation angle.** Footage filmed close to head-on (facing the camera) produced separation angles as low as 4°, even in frames that visually showed full hip-shoulder rotation. This happens because when the rotation axis is nearly parallel to the camera's line of sight, real 3D rotation compresses into very little apparent 2D motion.
→ *Fix:* Confirmed that a 90° side angle is optimal for all three metrics, since it keeps the rotation axis perpendicular to the camera — and standardized on that camera position going forward.

## 6. Current Data & Status

Data collection is ongoing. As of this writing, **30+ pitches are planned for capture and analysis** using the validated pipeline above; results will be logged to `pitching_analysis_result.csv` as they're processed.

| Pitch # | Velocity (mph) | Max Hip-Shoulder Separation (°) | Stride Length (m) | Release Extension (m) |
| :---: | :---: | :---: | :---: | :---: |
| *Pending* | *Updating...* | *Updating...* | *Updating...* | *Updating...* |

**Analysis plan:** Once a sufficient number of trials are collected, I plan to run a correlation/regression analysis between each kinematic variable and measured velocity, to test Hypotheses 1–2 quantitatively rather than just qualitatively.


## 7. Limitations & Next Steps

* **Single-pitch basis:** Each video currently yields one set of measurements; averaging across multiple pitches per session would reduce the effect of any single noisy measurement.
* **2D monocular limitation:** A single camera cannot fully capture depth, which can affect angle precision compared to a multi-camera or marker-based system.
* **Velocity correlation pending:** The kinematic pipeline is validated; the statistical analysis connecting it to measured velocity is the next phase of this project.
