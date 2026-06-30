# Methodology

This document explains how the analyzer measures each metric and why specific design decisions were made.

## Pose Detection

The program uses **MediaPipe Pose**, a machine learning model that detects 33 body landmarks (joints) from a single 2D video frame, including hips, shoulders, ankles, wrists, and elbows. Each landmark is returned as a set of (x, y, z) coordinates per frame.

Since a single video can contain multiple people (e.g., pitcher and catcher), the program filters detections to isolate the pitcher specifically — see "Identifying the Pitcher" below.

## Metric Definitions

### Hip-Shoulder Separation (HSS)
The angular difference between hip rotation and shoulder rotation at a given moment in the pitching motion. This is a key indicator of kinetic chain efficiency — elite pitchers generate more separation, which allows the upper body to "catch up" to the lower body and generate additional rotational velocity.

**Calculation:** Using the (x, y) coordinates of the left/right hip and left/right shoulder landmarks, the program computes the rotation angle of the hip line and shoulder line relative to a fixed reference axis, then takes the difference between the two.

**Camera angle requirement:** Because this is fundamentally a measurement of rotation, the camera must be positioned so the rotation axis is roughly perpendicular to the camera's line of sight (i.e., filming from the side / third-base line). Filming from behind the catcher places the rotation axis nearly parallel to the line of sight, which compresses the true 3D rotation into minimal apparent 2D movement, producing artificially low readings (see Debugging Journey in the main README).

### Stride Length
The horizontal distance between the pitcher's back foot (drive leg) and front foot (landing leg) at the moment of maximum extension during the stride.

**Frame selection:** The program does not simply search for the frame with maximum ankle-to-ankle distance, because this can incorrectly select a static standing position rather than a mid-pitch frame. Instead, it filters to frames where the throwing hand is positioned higher than the throwing shoulder — the point at which overhand pitchers reach full stride extension — before measuring ankle distance.

**Current limitation:** This filter is calibrated for overhand pitching motions. Submarine and three-quarter arm slot pitchers, whose throwing hand remains below the shoulder at full stride, are not yet correctly supported.

### Release Extension
The distance the throwing arm extends toward home plate at the moment of ball release, measured from the shoulder to the wrist landmark along the direction of throw.

## Known Limitations

1. **Pitching style assumption.** Current frame-filtering logic assumes an overhand delivery. Submarine and three-quarter pitchers are not yet supported.
2. **2D measurement constraints.** All current measurements are derived from 2D video, meaning accuracy depends heavily on camera placement relative to the motion being measured. A true 3D measurement (via stereo vision and camera calibration) would remove this dependency — this is a planned upgrade (see Roadmap in main README).
3. **Single-camera depth ambiguity.** Without depth information, any rotation or movement not roughly perpendicular to the camera's line of sight will be underestimated.

## Tech Stack

- **Python** — core program logic
- **MediaPipe** — pose landmark detection
- **OpenCV** — video frame extraction and processing
