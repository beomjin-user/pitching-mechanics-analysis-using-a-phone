"""
find_audio_offset.py — Find the time offset between two videos' audio
tracks via cross-correlation, instead of manually syncing on a clap.

Usage:
    python3 find_audio_offset.py video1.mov video2.mov

Requires ffmpeg on PATH (for audio extraction) and scipy.

Prints the offset in seconds: how much LATER video2's audio starts
relative to video1 (negative = video2 actually starts earlier).
That offset is what AUDIO_OFFSET_SEC in run_calibration.py expects.
"""

import sys
import subprocess
import tempfile
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate


SAMPLE_RATE = 16000  # downsampled — audio sync doesn't need hi-fi audio


def extract_audio(video_path, out_wav_path, sample_rate=SAMPLE_RATE):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "wav", out_wav_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video_path}:\n{result.stderr.decode()}")


def find_offset(video1_path, video2_path):
    with tempfile.TemporaryDirectory() as tmp:
        wav1 = os.path.join(tmp, "a1.wav")
        wav2 = os.path.join(tmp, "a2.wav")
        extract_audio(video1_path, wav1)
        extract_audio(video2_path, wav2)

        sr1, audio1 = wavfile.read(wav1)
        sr2, audio2 = wavfile.read(wav2)
        assert sr1 == sr2, f"Sample rate mismatch: {sr1} vs {sr2}"

        audio1 = audio1.astype(np.float64)
        audio2 = audio2.astype(np.float64)
        audio1 = (audio1 - audio1.mean()) / (audio1.std() + 1e-9)
        audio2 = (audio2 - audio2.mean()) / (audio2.std() + 1e-9)

        correlation = correlate(audio2, audio1, mode="full")
        lags = np.arange(-len(audio1) + 1, len(audio2))

        best_idx = int(np.argmax(correlation))
        best_lag_samples = lags[best_idx]
        offset_sec = best_lag_samples / sr1

        peak = correlation[best_idx]
        z_score = (peak - correlation.mean()) / (correlation.std() + 1e-9)

        return offset_sec, z_score


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 find_audio_offset.py <video1> <video2>")
        sys.exit(1)

    offset, z = find_offset(sys.argv[1], sys.argv[2])
    print(f"\nOffset: video2 starts {offset:+.3f} sec relative to video1")
    print(f"Confidence (z-score): {z:.1f}  (>5 is generally trustworthy)")
    if abs(z) < 5:
        print("WARNING: low confidence. Check that both videos actually share audio "
              "(ambient sound, glove pop, voices), or fall back to manually syncing "
              "on a visible/audible event instead.")
    print(f"\n-> Put this in run_calibration.py:  AUDIO_OFFSET_SEC = {offset:+.3f}")
