# Methodology

## Pipeline overview

1. **Pose extraction** — MediaPipe PoseLandmarker extracts 33 skeletal landmarks per frame from a single smartphone video.
2. **Motion phase classification** — Every frame is labeled into one of six pitching phases using a rule-based state machine driven by wrist speed and joint trajectory:

   ```
   windup → stride → cocking → acceleration → release → follow_through
   ```

3. **Metric extraction** — Each metric is measured only within the phase where it is physically meaningful (see below), rather than searching the whole clip for a maximum value.

## Metrics

### Hip-Shoulder Separation
Computed as the angular difference between the shoulder vector (landmarks 11, 12) and the hip vector (landmarks 23, 24), measured only in frames classified as `release`, and only when the throwing wrist is above the throwing shoulder.

### Stride Length
Ankle-to-ankle distance at the moment the lead foot lands, identified by a drop in vertical ankle velocity combined with a knee-flexion inflection point.

### Release Extension
Horizontal distance between the back foot (approximating the pitching rubber) and the throwing wrist at the moment the wrist reaches its farthest point toward home plate.

## Calibration

Pixel-to-metric scale is derived from the pitcher's known height (nose-to-ankle pixel distance), rather than an assumed shoulder-width constant. Release Extension is additionally adjusted using the wingspan-to-height ratio.

## Camera setup

Video is filmed from a single 90° side angle (3rd-base or 1st-base line), perpendicular to the pitching direction. This angle keeps the hip/shoulder rotation axis perpendicular to the camera, minimizing the angular compression that occurs from more frontal angles (see `validation.md` for why this matters).
