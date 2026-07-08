"""
app.py — Pitch Mechanics Analyzer Dashboard (Streamlit)

Usage:
    streamlit run app.py

Install:
    pip install streamlit

This UI wraps analyze_stereo_hss.py into a clean visual dashboard.
No terminal commands needed — just select videos and click Analyze.
"""

import streamlit as st
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from find_audio_offset import find_offset

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pitch Mechanics Analyzer",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* dark sport-analytics palette */
  .stApp { background: #0d1117; color: #e6edf3; }

  /* sidebar */
  [data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] label { color: #8b949e !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }

  /* metric cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    transition: border-color .2s;
  }
  .metric-card:hover { border-color: #58a6ff; }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #58a6ff;
    line-height: 1;
    margin-bottom: 0.3rem;
  }
  .metric-label {
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
  }

  /* pitch row */
  .pitch-row {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 2rem;
  }
  .pitch-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #58a6ff;
    min-width: 2.5rem;
  }
  .pitch-stat { font-size: 0.88rem; color: #c9d1d9; }
  .pitch-stat span { color: #e6edf3; font-weight: 600; }

  /* warning badge */
  .warn-badge {
    background: #2d1b00;
    border: 1px solid #d29922;
    color: #d29922;
    border-radius: 4px;
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    margin-left: 0.5rem;
  }

  /* section headers */
  .section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8b949e;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #30363d;
    padding-bottom: 0.4rem;
  }

  /* status bar */
  .status-ok  { color: #3fb950; font-weight: 600; }
  .status-warn{ color: #d29922; font-weight: 600; }

  /* image captions */
  .img-caption {
    font-size: 0.72rem;
    color: #8b949e;
    text-align: center;
    margin-top: 0.3rem;
  }

  /* hide default streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── helpers ──────────────────────────────────────────────────────────────────

def list_videos(folder):
    exts = {".mov", ".mp4", ".MOV", ".MP4"}
    return sorted([p for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix in exts])


def parse_summary(summary_path):
    """Parse summary.txt into a list of pitch dicts + metadata dict."""
    pitches, meta = [], {}
    try:
        lines = Path(summary_path).read_text().splitlines()
        for line in lines:
            if line.startswith("P") and "|" in line:
                parts = [x.strip() for x in line.split("|")]
                if len(parts) >= 4:
                    try:
                        p_num = int(parts[0].replace("P", "").strip())
                        hss   = parts[1].strip()
                        stride = parts[2].strip()
                        ext    = parts[3].strip()
                        pitches.append({
                            "n": p_num,
                            "hss": hss, "stride": stride, "ext": ext,
                            "high_spread": "rough" in line.lower()
                        })
                    except Exception:
                        pass
            for key, attr in [
                ("Camera baseline", "baseline"),
                ("Consistency check", "consistency"),
                ("Matched frames", "matched"),
                ("Audio offset", "offset"),
                ("cam1 FOV", "fov1"),
                ("cam2 FOV", "fov2"),
            ]:
                if line.startswith(key):
                    meta[attr] = line.split(":", 1)[-1].strip()
    except Exception:
        pass
    return pitches, meta


def find_latest_result(cam1_stem):
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    candidates = sorted(
        [d for d in results_dir.iterdir()
         if d.is_dir() and d.name.startswith(cam1_stem)],
        key=lambda d: d.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


# ─── sync-check helpers ────────────────────────────────────────────────────────
# These render a handful of candidate cam1/cam2 offsets around the automatic
# audio-based estimate so you can visually confirm (or correct) alignment
# BEFORE any calibration/HSS math runs. A confident audio z-score only means
# the waveforms line up well - it doesn't guarantee frame-perfect alignment,
# and a couple of frames of residual error is enough to show e.g. the ball
# still in hand on one camera while it's already left the hand on the other,
# which quietly corrupts every downstream stereo computation.

SYNC_N_EACH_SIDE = 3
SYNC_TARGET_H = 360


def grab_frame(video_path, t_sec, fps):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t_sec * fps))))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def resize_to_height(frame, target_h):
    h, w = frame.shape[:2]
    scale = target_h / h
    return cv2.resize(frame, (int(round(w * scale)), target_h))


def find_release_time_for_sync(cam1_stem):
    """
    Uses the release-frame timestamp analyze_stereo_hss.py already computes
    per pitch (results/<cam1_stem>_*/pitches.json) - the same definition
    used to draw P{n}_release.png (wrist farthest forward while the arm is
    raised, near foot plant). That's a better-grounded "sharp, unambiguous
    moment" than any fresh ad-hoc heuristic, and it's already validated by
    the fact that it produces sensible-looking release images.

    Note: pitches.json's release_time_sec is derived purely from cam1's own
    landmarks, so it stays valid even if the run it came from used a wrong
    cam2 pairing/offset - only the offset itself needs re-checking here.

    Returns None if no prior run exists yet for this cam1 video, or none of
    its detected pitches had a usable release frame.
    """
    result_dir = find_latest_result(cam1_stem)
    if result_dir is None:
        return None
    pitches_path = result_dir / "pitches.json"
    if not pitches_path.exists():
        return None
    pitches = json.loads(pitches_path.read_text())
    for p in pitches:
        if p.get("release_time_sec") is not None:
            return p["release_time_sec"]
    return None


def build_sync_candidates(cam1_path, cam2_path, approx_t,
                           n_each_side=SYNC_N_EACH_SIDE, target_h=SYNC_TARGET_H):
    """Returns (candidates, auto_offset, z) where candidates is a list of
    {"offset": float, "image": RGB numpy array, "is_auto": bool}."""
    cap1 = cv2.VideoCapture(cam1_path)
    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30.0
    cap1.release()
    cap2 = cv2.VideoCapture(cam2_path)
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30.0
    cap2.release()
    step = 1.0 / fps2

    auto_offset, z = find_offset(cam1_path, cam2_path)

    candidates = []
    for i in range(-n_each_side, n_each_side + 1):
        cand_offset = auto_offset + i * step
        fr1 = grab_frame(cam1_path, approx_t, fps1)
        fr2 = grab_frame(cam2_path, approx_t + cand_offset, fps2)
        if fr1 is None or fr2 is None:
            continue
        fr1 = resize_to_height(fr1, target_h)
        fr2 = resize_to_height(fr2, target_h)
        combined = cv2.hconcat([fr1, fr2])
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        candidates.append({
            "offset": cand_offset,
            "image": combined_rgb,
            "is_auto": (i == 0),
        })
    return candidates, auto_offset, z


# ─── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚾ Pitch Mechanics")
    st.markdown("---")

    folder = st.text_input("Video folder", value=".", help="Folder to scan for .mov / .mp4 files")
    videos = list_videos(folder)
    video_names = [v.name for v in videos]

    if not video_names:
        st.warning("No video files found in this folder.")
        st.stop()

    st.markdown('<div class="section-label">Side-view camera (3rd-base line)</div>', unsafe_allow_html=True)
    cam1_name = st.selectbox("cam1", video_names, key="cam1", label_visibility="collapsed")

    st.markdown('<div class="section-label">Back-view camera (behind catcher)</div>', unsafe_allow_html=True)
    other_names = [n for n in video_names if n != cam1_name]
    if not other_names:
        st.error("Need at least 2 video files.")
        st.stop()
    cam2_name = st.selectbox("cam2", other_names, key="cam2", label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
    skip_extract = st.checkbox(
        "Skip MediaPipe extraction",
        help="Reuse existing JSON files if the same videos were already analyzed. Much faster.")

    st.markdown("---")
    run_btn = st.button("▶  Analyze", use_container_width=True, type="primary")


# ─── main area ────────────────────────────────────────────────────────────────

st.markdown("# Pitch Mechanics Analyzer")
st.markdown(
    f"<span style='color:#8b949e;font-size:0.85rem'>"
    f"cam1 · <code>{cam1_name}</code> &nbsp;|&nbsp; cam2 · <code>{cam2_name}</code>"
    f"</span>", unsafe_allow_html=True)

cam1_path = str(Path(folder) / cam1_name)
cam2_path = str(Path(folder) / cam2_name)
cam1_stem = Path(cam1_name).stem

# ── Sync Check ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Sync Check (recommended before analyzing)</div>',
            unsafe_allow_html=True)
st.markdown(
    "<span style='color:#8b949e;font-size:0.82rem'>"
    "A confident audio sync can still be off by a frame or two. This finds "
    "the fastest moment in the video (almost always right around a pitch "
    "release) automatically and shows both cameras there, so you can "
    "confirm - or fix - the alignment without hunting for a timestamp."
    "</span>", unsafe_allow_html=True)

if st.button("🔍  동기화 확인 시작 (릴리즈 시점 자동 감지)"):
    with st.spinner("이전 분석 결과에서 release 시점 가져오는 중..."):
        auto_t = find_release_time_for_sync(cam1_stem)
    if auto_t is None:
        st.warning(
            "이 cam1 영상은 아직 분석된 적이 없어서 release 시점을 가져올 수 없어. "
            "▶ Analyze를 한 번 돌려서 pitches.json을 만든 다음 다시 시도하거나, "
            "아래 '직접 시각 입력'에서 대략적인 릴리즈 시점을 눈대중으로 넣어줘.")
        auto_t = 1.0
    with st.spinner("Building sync candidates..."):
        candidates, auto_offset, z = build_sync_candidates(cam1_path, cam2_path, auto_t)
    if not candidates:
        st.error("프레임을 못 읽었어 — 영상 파일 자체를 확인해줘.")
    else:
        st.session_state["sync_candidates"] = candidates
        st.session_state["sync_idx"] = len(candidates) // 2  # start at auto offset
        st.session_state["sync_approx_t"] = auto_t

if st.session_state.get("sync_candidates"):
    candidates = st.session_state["sync_candidates"]
    idx = max(0, min(st.session_state.get("sync_idx", len(candidates) // 2),
                      len(candidates) - 1))
    cand = candidates[idx]

    st.image(cand["image"], use_container_width=True)
    tag = " (auto-detected)" if cand["is_auto"] else ""
    st.markdown(
        f"<span style='color:#8b949e'>Candidate {idx + 1}/{len(candidates)}"
        f" &nbsp;·&nbsp; offset {cand['offset']:+.4f}s{tag}"
        f" &nbsp;·&nbsp; 기준 시각(자동 감지): {st.session_state['sync_approx_t']:.2f}s</span>",
        unsafe_allow_html=True)

    c_prev, c_next, c_pick = st.columns([1, 1, 2])
    with c_prev:
        if st.button("◀ 이전", disabled=(idx == 0), use_container_width=True):
            st.session_state["sync_idx"] = idx - 1
            st.rerun()
    with c_next:
        if st.button("다음 ▶", disabled=(idx == len(candidates) - 1), use_container_width=True):
            st.session_state["sync_idx"] = idx + 1
            st.rerun()
    with c_pick:
        if st.button("✓  이 프레임이 맞음 — 분석 시작", type="primary", use_container_width=True):
            with open("sync_offset.json", "w") as f:
                json.dump({
                    "offset_sec": cand["offset"],
                    "cam1_video": cam1_path,
                    "cam2_video": cam2_path,
                }, f, indent=2)
            st.session_state["auto_run_after_sync"] = True
            st.rerun()

    with st.expander("자동 감지된 순간이 이상해 보이면 - 직접 시각 입력"):
        manual_t = st.number_input(
            "기준 시각 (초, cam1 기준)", min_value=0.0,
            value=float(st.session_state["sync_approx_t"]), step=0.05, format="%.2f")
        if st.button("이 시각으로 다시 만들기"):
            with st.spinner("Rebuilding..."):
                candidates2, _, _ = build_sync_candidates(cam1_path, cam2_path, manual_t)
            if candidates2:
                st.session_state["sync_candidates"] = candidates2
                st.session_state["sync_idx"] = len(candidates2) // 2
                st.session_state["sync_approx_t"] = manual_t
                st.rerun()
            else:
                st.error("프레임을 못 읽었어 — 시각이 두 영상 길이 안에 있는지 확인해줘.")

st.markdown("---")


# ─── run analysis ─────────────────────────────────────────────────────────────

if run_btn or st.session_state.pop("auto_run_after_sync", False):
    cmd = [sys.executable, "analyze_stereo_hss.py", cam1_path, cam2_path]
    if skip_extract:
        cmd.append("--skip-extract")

    st.markdown("---")
    log_box  = st.empty()
    progress = st.progress(0)
    logs = []

    STEP_MARKERS = {
        "[1/5]": 0.05,
        "[2/5]": 0.15,
        "[3/5]": 0.30,
        "[4/5]": 0.80,
        "[5/5]": 0.95,
    }

    with st.spinner("Running analysis…"):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logs.append(line)
                log_box.code("\n".join(logs[-30:]), language="")
                for marker, pct in STEP_MARKERS.items():
                    if marker in line:
                        progress.progress(pct)
        proc.wait()

    progress.progress(1.0)
    if proc.returncode == 0:
        st.success("Analysis complete!")
    else:
        st.error("Analysis failed — see log above.")

    time.sleep(0.5)
    st.rerun()


# ─── show results ─────────────────────────────────────────────────────────────

result_dir = find_latest_result(cam1_stem)

if result_dir is None:
    st.markdown("---")
    st.info("Select two videos, then click **▶ Analyze** to get started.")
    st.stop()

summary_path = result_dir / "summary.txt"
pitches, meta = parse_summary(summary_path)

if not pitches and not meta:
    st.info("No results yet. Click **▶ Analyze** to run.")
    st.stop()

# run timestamp from folder name
folder_ts = result_dir.name.replace(cam1_stem + "_", "")
st.markdown(
    f"<span style='color:#8b949e;font-size:0.8rem'>Last run: {folder_ts} &nbsp;·&nbsp; {result_dir}</span>",
    unsafe_allow_html=True)

st.markdown("---")

# ── metadata row ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Session Stats</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

def meta_card(col, label, value):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>', unsafe_allow_html=True)

meta_card(m1, "Pitches detected",   str(len(pitches)))
meta_card(m2, "Matched frames",     meta.get("matched", "—"))
meta_card(m3, "Camera baseline",    meta.get("baseline", "—"))
consistency = meta.get("consistency", "")
cons_color  = "#3fb950" if consistency.startswith("OK") else "#d29922"
m4.markdown(
    f'<div class="metric-card">'
    f'<div class="metric-value" style="color:{cons_color};font-size:1.4rem">'
    f'{"✓" if consistency.startswith("OK") else "⚠"}</div>'
    f'<div class="metric-label">Consistency</div>'
    f'<div style="font-size:0.7rem;color:#8b949e;margin-top:0.3rem">{consistency}</div>'
    f'</div>', unsafe_allow_html=True)

st.markdown("---")

# ── HSS over time plot ────────────────────────────────────────────────────────
hss_plot = result_dir / "hss_over_time.png"
if hss_plot.exists():
    st.markdown('<div class="section-label">Hip-Shoulder Separation Over Time</div>', unsafe_allow_html=True)
    st.image(str(hss_plot), use_container_width=True)
    st.markdown("---")

# ── per-pitch section ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Per-Pitch Results</div>', unsafe_allow_html=True)

if not pitches:
    st.warning("No pitches were detected. Try running the analysis again.")
    st.stop()

for p in pitches:
    warn = '<span class="warn-badge">rough estimate</span>' if p["high_spread"] else ""
    st.markdown(
        f'<div class="pitch-row">'
        f'<span class="pitch-num">P{p["n"]}</span>'
        f'<span class="pitch-stat">HSS <span>{p["hss"]}°</span></span>'
        f'<span class="pitch-stat">Stride <span>{p["stride"]} m</span></span>'
        f'<span class="pitch-stat">Extension <span>{p["ext"]} m</span></span>'
        f'{warn}'
        f'</div>',
        unsafe_allow_html=True)

st.markdown("---")

# ── image gallery ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Frame Gallery</div>', unsafe_allow_html=True)

pitch_nums = [p["n"] for p in pitches]
selected_n = st.select_slider("Pitch", options=pitch_nums, value=pitch_nums[0])

hss_img = result_dir / f"P{selected_n}_hss.png"
rel_img = result_dir / f"P{selected_n}_release.png"

col_hss, col_rel = st.columns(2)

with col_hss:
    if hss_img.exists():
        st.image(str(hss_img), use_container_width=True)
        st.markdown('<div class="img-caption">Max Hip-Shoulder Separation · just after foot plant</div>',
                    unsafe_allow_html=True)
    else:
        st.info("HSS frame not available for this pitch.")

with col_rel:
    if rel_img.exists():
        st.image(str(rel_img), use_container_width=True)
        st.markdown('<div class="img-caption">Release Point · stride line + extension</div>',
                    unsafe_allow_html=True)
    else:
        st.info("Release frame not available for this pitch.")

# ── 더 보기: 오버레이 영상 ──────────────────────────────────────────────────────
overlay_path = result_dir / f"P{selected_n}_overlay.mp4"

st.markdown("---")
st.markdown('<div class="section-label">Overlay Video</div>', unsafe_allow_html=True)

if overlay_path.exists():
    # 더 보기 토글 버튼
    show_key = f"show_overlay_{selected_n}"
    if show_key not in st.session_state:
        st.session_state[show_key] = False

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        if st.button(
            "▶  더 보기" if not st.session_state[show_key] else "✕  닫기",
            key=f"btn_overlay_{selected_n}",
        ):
            st.session_state[show_key] = not st.session_state[show_key]
    with col_hint:
        st.markdown(
            "<span style='color:#8b949e;font-size:0.82rem;line-height:2.4'>"
            "투구 구간 전체를 재생하면서 HSS / Stride / Extension이 실시간으로 표시됩니다."
            "</span>", unsafe_allow_html=True)

    if st.session_state[show_key]:
        with open(str(overlay_path), "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.markdown(
            f'<div class="img-caption">P{selected_n} · {overlay_path.name}'
            f'  ·  {overlay_path.stat().st_size // 1024} KB</div>',
            unsafe_allow_html=True)
else:
    st.markdown(
        "<span style='color:#8b949e;font-size:0.82rem'>"
        "오버레이 영상이 없습니다. "
        "<code>pitch_visualizer.py</code>로 먼저 분석을 실행하세요."
        "</span>", unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<span style='color:#484f58;font-size:0.72rem'>"
    "Pitch Mechanics Analyzer · Computer Vision Project · "
    "MediaPipe + OpenCV + Stereo Self-Calibration"
    "</span>", unsafe_allow_html=True)