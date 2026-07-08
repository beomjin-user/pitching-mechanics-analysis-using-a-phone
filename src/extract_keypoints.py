"""
extract_keypoints.py — Run this LOCALLY on your machine (not here in chat)
on each of your two synced phone videos, to extract MediaPipe pose
keypoints per frame.

Usage:
    python3 extract_keypoints.py path/to/video.mov path/to/output.json

Produces a JSON file:
{
  "video_path": "...",
  "fps": 30.0,
  "image_width": 1080,
  "image_height": 1920,
  "frames": [
    {"frame_idx": 0, "timestamp_sec": 0.0,
     "landmarks": [{"x":.., "y":.., "z":.., "visibility":..}, ... 33 total ...]},
    ...
  ]
}

Coordinates in "landmarks" are PIXEL coordinates (MediaPipe's normalized
0-1 output multiplied back by image width/height) — geometry.py expects
pixel coordinates and does its own FOV-based normalization, so don't
pre-normalize here.

Requires the same pose_landmarker_heavy.task model file
pitch_summary_v2.py already uses, in the same folder (or edit MODEL_PATH).
"""

import sys
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_heavy.task"


def bbox_area(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def extract(video_path, output_path, model_path=MODEL_PATH):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=3,  # detect up to 3 people; filtered down to one below
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_out = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                # Same heuristic as pitch_summary_v2.py: in catch-play
                # footage, the pitcher is usually the smallest bounding box
                # (farthest from camera). If your two-camera setup instead
                # frames ONLY the pitcher, this is a no-op (only one pose).
                chosen = min(result.pose_landmarks, key=bbox_area)

                landmarks_out = [
                    {"x": lm.x * width, "y": lm.y * height, "z": lm.z, "visibility": lm.visibility}
                    for lm in chosen
                ]
                frames_out.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                    "landmarks": landmarks_out,
                })

            frame_idx += 1

    cap.release()

    with open(output_path, "w") as f:
        json.dump({
            "video_path": video_path,
            "fps": fps,
            "image_width": width,
            "image_height": height,
            "frames": frames_out,
        }, f)

    print(f"Saved {len(frames_out)} frames with detected pose -> {output_path}")
    print(f"(out of frames read; frames with no confident pose detection are skipped)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_keypoints.py <video_path> <output_json_path>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
