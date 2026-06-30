# Future Work

## Near-term

- **Multi-pitch averaging:** Currently each video yields a single measurement. Collecting 30+ pitches per session and averaging would reduce the impact of any single noisy frame.
- **Velocity correlation:** A radar gun reading is logged per pitch, but the statistical analysis connecting kinematic metrics (Hip-Shoulder Separation, Stride Length, Release Extension) to measured velocity has not yet been run. Planned: regression analysis once enough trials are collected.

## Possible upgrades

- **3D measurement via stereo vision:** The current pipeline measures hip-shoulder separation in 2D from a single camera. A true 3D measurement (via two synchronized cameras, camera calibration, and triangulation) would remove the remaining angular distortion that even an optimal 90° camera angle doesn't fully eliminate. This is the most likely next upgrade — currently weighing it against MediaPipe's built-in `pose_world_landmarks` depth estimates, which would be a smaller change but a less accurate one.
- **Left-handed pitcher validation:** The pipeline supports a left-handed configuration, but it hasn't been tested on actual left-handed pitching footage yet.

## Known limitations

See `validation.md` for issues already found and fixed. The main remaining limitation is the 2D/monocular nature of the current camera setup.
