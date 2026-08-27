# Smartphone-Based Pitching Biomechanics Analysis

**Braylon** · St. Mark’s School, Class of 2027  
braylonkim@stmarksschool.org

---

## Overview

As a varsity baseball pitcher stuck in a velocity plateau, I wanted objective biomechanical feedback without access to expensive motion-capture systems (Rapsodo, TrackMan, etc.). I built a computer vision pipeline that measures three key indicators of pitching efficiency from ordinary smartphone video:

- **Hip-Shoulder Separation**
- **Stride Length**
- **Release Extension**

The system uses two synchronized phone cameras, MediaPipe pose estimation, and 3D triangulation.

---

## Experimental Setup

- **Cameras:** Two iPhones (slow-motion or regular mode, 30–240 fps) placed at ~90° side angles
- **Software:** Python, OpenCV, MediaPipe PoseLandmarker, Streamlit
- **Calibration:** Pitcher’s known height (nose-to-ankle) + wingspan adjustment
- **Ground truth:** Radar gun for velocity

The pipeline first syncs the two videos (audio + visual check), recovers relative camera pose via Essential Matrix, then triangulates 3D joint positions before computing metrics.

![Camera Setup](https://github.com/user-attachments/assets/52544ad7-465c-48d4-99e4-dddd8fc180db)

*Two-camera side-view setup used for 3D triangulation*

---

## Motivation & Hypotheses

Elite pitchers generate higher velocity largely through better use of the kinetic chain. I hypothesized that:

1. Larger hip-shoulder separation near release correlates with higher velocity
2. Greater stride length and release extension (relative to height) also correlate with higher velocity

Velocity correlation analysis is still in progress.

---

## Key Technical Challenges Solved

| Problem | Solution |
|---------|----------|
| Catcher misidentified as pitcher | Select the *smallest* person in the frame |
| Static stance counted as a pitch | Require throwing wrist above shoulder + minimum rotational velocity |
| Single-camera angle severely under-estimated separation | Switched to stereo 3D triangulation |
| Peak separation jumped due to occlusion noise | Smooth the angle curve before finding the max |
| Browser-unplayable overlay videos | Re-encode with ffmpeg (H.264) |

---

## Current Status & Next Steps

The system currently outputs per-pitch measurements, time-series plots of hip-shoulder separation, and tracked overlay videos through a Streamlit dashboard.

**Still to do:**
- Pair kinematic data with radar velocity for correlation analysis
- Improve pitch detection beyond current heuristics
- Make field-of-view calibration phone-model specific
- Extend 3D triangulation to stride length and release extension

---
