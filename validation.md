# Validation

This document tracks the debugging and validation process — most of the actual difficulty in this project was here, not in writing the first version of the pipeline.

## 1. Wrong person identified as the pitcher

In footage filmed with another person near the camera (e.g. a catch partner), MediaPipe sometimes locked onto the closer/larger person instead of the pitcher, producing biomechanically impossible values (e.g. 80+ degree hip-shoulder separation).

**Fix:** When multiple people are detected in frame, the smallest (farthest from camera) person is automatically selected as the pitcher.

## 2. Static stance misidentified as the pitching motion

Early versions searched the full clip for the frame with the largest stride distance or rotation angle. This sometimes picked a relaxed standing pose (legs spread, not mid-pitch) instead of an actual pitch.

**Fix:** Measurements are now restricted to frames where the throwing wrist is positioned above the throwing shoulder — a condition that is only true during an actual cocking/release motion, not a resting stance.

## 3. Camera angle distorting the separation angle

Footage filmed close to head-on (facing the camera) produced separation angles as low as 4°, even in frames that visually showed full hip-shoulder rotation. When the rotation axis is nearly parallel to the camera's line of sight, real 3D rotation compresses into very little apparent 2D motion.

**Fix:** Confirmed that a 90° side angle is optimal for all three metrics, and standardized on that camera position for all future recordings.

## Verification method

For every measurement, the pipeline also saves an overlay image of the exact frame used (joints + measurement line drawn on the original video frame), so each number can be visually checked against the source footage rather than trusted blindly. See `figures/` for examples.
