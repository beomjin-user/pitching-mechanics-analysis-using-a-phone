# Results & Validation

This document tracks how accurately the program measures each metric, compared against manual/ground-truth measurements.

> **Note:** Fill in the sections below with actual measured data. Quantitative validation is what separates this from a hobby script — it shows the metrics are not just computed, but verified.

## Validation Method

Describe how you checked accuracy. For example:
- Manually measured hip-shoulder separation angle frame-by-frame in a video editor and compared to the program's output
- Compared stride length against a tape-measured distance on the field
- Re-ran the same pitch video multiple times to check for consistency

## Hip-Shoulder Separation Accuracy

| Trial | Program Output | Manual/Ground Truth | Error |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

**Average error:** ___ degrees
**Notes:** (e.g., does accuracy change with camera distance, lighting, pitcher's clothing, etc.?)

## Stride Length Accuracy

| Trial | Program Output | Manual/Ground Truth | Error |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

**Average error:** ___ inches/cm

## Release Extension Accuracy

| Trial | Program Output | Manual/Ground Truth | Error |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

**Average error:** ___ inches/cm

## Camera Angle Sensitivity

Document how much accuracy degrades depending on camera placement. For example, testing the same pitch filmed from:
- Directly behind catcher
- 45-degree angle
- Full side view (third-base line)

| Camera Angle | HSS Reading | Error vs. Side View |
|---|---|---|
| Behind catcher | | |
| 45 degrees | | |
| Side view (baseline) | | |

## Known Failure Cases

List specific situations where the program currently produces inaccurate or unusable results (e.g., submarine pitching motion, poor lighting, partial occlusion of landmarks).

## Next Steps for Improving Accuracy

- [ ] Stereo vision upgrade for true 3D triangulation
- [ ] Support for non-overhand pitching motions
- [ ] Larger validation dataset across multiple pitchers
