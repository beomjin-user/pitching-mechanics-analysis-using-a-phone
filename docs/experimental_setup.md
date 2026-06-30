# Experimental Setup

## Hardware

- **Camera:** iPhone, slow-motion mode (60–240fps depending on device)
- **Camera position:** 90° side angle relative to the pitching direction (3rd-base or 1st-base line), perpendicular to the rotation axis
- **Velocity reference:** Radar gun (used to log actual pitch velocity alongside each video for future correlation analysis)

## Software

- Python 3.10
- MediaPipe PoseLandmarker (heavy model)
- OpenCV
- Pandas / NumPy
- Matplotlib

## Subject

- Height and wingspan are entered manually and used for pixel-to-metric scale calibration (see `methodology.md`).
- Right/left-handedness is configured in the script to determine which side's joints represent the "throwing" arm/leg.

## Data collection status

Video collection is ongoing — see `future_work.md` for the planned sample size and next analysis steps.
