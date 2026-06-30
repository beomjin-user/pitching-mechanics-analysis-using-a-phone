import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# ──────────────────────────────────────────
# 설정 — 여기만 수정하면 됩니다
# ──────────────────────────────────────────
# 측면(3루 또는 1루쪽 90도) 영상 하나로 HSS, 보폭, 익스텐션을 전부 측정한다.
# 이 각도가 세 지표 모두에 최적이라(골반/어깨 회전이 카메라와 수직으로 보임),
# 더 이상 후면 영상을 따로 둘 필요가 없다.
#
# VIDEO_PATH를 고정하지 않고, 실행할 때마다 현재 폴더에 있는 영상 파일들을
# 자동으로 찾아 번호로 선택하게 한다(아래 select_video_interactively 참고).
# 새 영상을 찍으면 그냥 이 폴더에 옮겨놓기만 하면 되고, 코드를 열어 파일명을
# 고칠 필요가 없다.
VIDEO_SEARCH_DIR = "."                   # 영상을 찾을 폴더 (기본: 현재 폴더)
VIDEO_EXTENSIONS  = [".mov", ".mp4", ".MOV", ".MP4"]

MODEL_PATH       = "pose_landmarker_heavy.task"
PLAYER_HEIGHT_M  = 1.83                  # 본인 키 (미터)
PLAYER_WINGSPAN_M = 1.93                 # 본인 윙스팬 (양팔을 벌렸을 때 손끝-손끝 거리, 미터)
THROWS_RIGHT     = True                  # 오른손 투수 True / 왼손 False

# 화면에 투수 외 다른 사람(포수, 관찰자 등)이 같이 잡히는 영상이면 2 이상으로 설정.
# 캐치볼처럼 카메라 바로 앞에 사람이 서 있는 구도에서 특히 필요.
NUM_POSES         = 2
# "smallest": 화면에서 가장 작게(=카메라에서 가장 멀리) 보이는 사람을 투수로 선택.
#   카메라 앞사람이 투수보다 가까이 있어 크게 잡히는 일반적인 캐치볼 구도에 적합.
PITCHER_SELECT_MODE = "smallest"

# 비정상치 판정 기준 (단계 분류와 별개로, 최종 안전장치로 유지)
STRIDE_RATIO_MIN, STRIDE_RATIO_MAX = 0.3, 1.2
HSS_MIN, HSS_MAX = 0, 90

IDX = {
    "nose":       0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow":    13, "r_elbow":    14,
    "l_wrist":    15, "r_wrist":    16,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
}

PHASES = ["windup", "stride", "cocking", "acceleration", "release", "follow_through"]
PHASE_COLORS_BGR = {
    "windup":         (180, 180, 180),
    "stride":         (255, 165, 0),
    "cocking":        (0, 200, 255),
    "acceleration":   (60, 60, 230),
    "release":        (0, 255, 0),
    "follow_through": (200, 100, 200),
}
PHASE_COLORS_HEX = {
    "windup":         "#888780",
    "stride":         "#0F6E56",
    "cocking":        "#BA7517",
    "acceleration":   "#D85A30",
    "release":        "#1D9E75",
    "follow_through": "#534AB7",
}


# ──────────────────────────────────────────
# 공용 유틸
# ──────────────────────────────────────────
def lm_to_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h])


def axis_angle(p_left, p_right):
    dx = p_right[0] - p_left[0]
    dy = p_right[1] - p_left[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def dist(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def create_landmarker(model_path, mode, num_poses=1):
    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mode,
        num_poses=num_poses,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return PoseLandmarker.create_from_options(options)


def select_pitcher_pose(pose_landmarks_list, mode="smallest"):
    """
    여러 사람이 감지됐을 때 투수를 자동으로 선택한다.

    mode="smallest": 어깨너비가 가장 작은(=화면에서 가장 작게, 즉 카메라에서
                      가장 멀리 있는) 사람을 투수로 간주한다.
                      캐치볼처럼 카메라 바로 앞에 다른 사람(포수/관찰자)이
                      서 있는 구도에서, 투수는 보통 더 멀리 있어 더 작게 잡힌다.

    반환값: 선택된 사람의 landmark 리스트, 또는 감지된 사람이 없으면 None
    """
    if not pose_landmarks_list:
        return None
    if len(pose_landmarks_list) == 1:
        return pose_landmarks_list[0]

    if mode == "smallest":
        best = None
        best_size = None
        for lms in pose_landmarks_list:
            try:
                l_sh = lms[11]
                r_sh = lms[12]
                w = abs(l_sh.x - r_sh.x)
                h = abs(l_sh.y - r_sh.y)
                size = (w**2 + h**2) ** 0.5
            except (IndexError, AttributeError):
                continue
            if best_size is None or size < best_size:
                best_size = size
                best = lms
        return best if best is not None else pose_landmarks_list[0]

    return pose_landmarks_list[0]


# ──────────────────────────────────────────
# 관절 좌표 전체 추출 (한 번만 돌리고 재사용)
# ──────────────────────────────────────────
def extract_all_landmarks(video_path, model_path, throws_right, num_poses=1, pitcher_select="smallest"):
    """
    영상의 모든 프레임에서 필요한 관절 좌표를 추출해 DataFrame으로 반환.
    이후 단계 분류, HSS 계산, 보폭 계산이 전부 이 DataFrame 하나로 처리된다.

    num_poses > 1로 설정하면 화면에 여러 사람이 있어도 각각 인식한 뒤
    select_pitcher_pose()로 투수 한 명만 골라서 사용한다.
    캐치볼처럼 카메라 앞에 다른 사람(포수/관찰자)이 같이 잡히는 영상에서 사용.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_factor = 3 if w > 1500 else 1
    rows = []

    throw_wrist = "r_wrist" if throws_right else "l_wrist"
    throw_elbow = "r_elbow" if throws_right else "l_elbow"
    throw_shoulder = "r_shoulder" if throws_right else "l_shoulder"
    front_ankle = "l_ankle" if throws_right else "r_ankle"
    back_ankle  = "r_ankle" if throws_right else "l_ankle"
    lead_knee   = "l_knee" if throws_right else "r_knee"

    with create_landmarker(model_path, RunningMode.VIDEO, num_poses=num_poses) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            proc = cv2.resize(frame, (w // scale_factor, h // scale_factor)) if scale_factor > 1 else frame
            ph, pw = proc.shape[:2]

            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            row = {"frame": frame_idx, "time_sec": frame_idx / fps, "detected": False}

            lms = select_pitcher_pose(result.pose_landmarks, mode=pitcher_select) if result.pose_landmarks else None

            if lms is not None:
                avg_vis = np.mean([lms[i].visibility for i in IDX.values()])

                if avg_vis > 0.3:
                    row["detected"] = True
                    row["avg_vis"] = avg_vis
                    for name, idx in IDX.items():
                        xy = lm_to_xy(lms[idx], pw, ph)
                        row[f"{name}_x"] = xy[0]
                        row[f"{name}_y"] = xy[1]
                        row[f"{name}_vis"] = lms[idx].visibility

                    row["shoulder_w_px"] = dist(
                        lm_to_xy(lms[IDX["l_shoulder"]], pw, ph),
                        lm_to_xy(lms[IDX["r_shoulder"]], pw, ph),
                    )
                    # 분류에 쓸 핵심 변수들을 미리 표준 이름으로도 저장
                    row["throw_wrist_x"] = lm_to_xy(lms[IDX[throw_wrist]], pw, ph)[0]
                    row["throw_wrist_y"] = lm_to_xy(lms[IDX[throw_wrist]], pw, ph)[1]
                    row["throw_shoulder_y"] = lm_to_xy(lms[IDX[throw_shoulder]], pw, ph)[1]
                    row["lead_knee_y"]   = lm_to_xy(lms[IDX[lead_knee]], pw, ph)[1]
                    row["front_ankle_x"] = lm_to_xy(lms[IDX[front_ankle]], pw, ph)[0]
                    row["front_ankle_y"] = lm_to_xy(lms[IDX[front_ankle]], pw, ph)[1]
                    row["back_ankle_x"]  = lm_to_xy(lms[IDX[back_ankle]], pw, ph)[0]
                    row["back_ankle_y"]  = lm_to_xy(lms[IDX[back_ankle]], pw, ph)[1]
                    row["throw_wrist_vis"] = lms[IDX[throw_wrist]].visibility

            rows.append(row)
            frame_idx += 1

    cap.release()
    df = pd.DataFrame(rows)
    print(f"  영상: {w}x{h} / {fps:.1f}fps / {total}프레임, 감지 {df['detected'].sum()}/{total}")
    return df, fps


# ──────────────────────────────────────────
# 동작 단계 자동 분류 (규칙 기반 상태머신)
# ──────────────────────────────────────────
def classify_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    손목 속도, 무릎 높이, 발목 거리 패턴으로 6단계를 자동 분류한다.

    분류 로직:
        1) 투구손 손목의 프레임간 이동속도(throw_wrist_speed)를 계산
        2) 속도가 전체 최대치에 도달하는 프레임을 "release 중심"으로 잡음
        3) release 중심 앞뒤로 속도 변화 패턴을 따라가며 6단계 경계를 정함
           - windup: 영상 시작 ~ 리드 무릎이 가장 높이 올라간 시점
           - stride: 무릎 최고점 ~ 앞발이 착지(발목 y 최대화 + 속도 급감)
           - cocking: 착지 ~ 손목 속도가 본격적으로 증가하기 시작하는 시점
           - acceleration: 속도 증가 구간 ~ release 중심 직전
           - release: release 중심 ± 작은 윈도우 (속도 피크 부근)
           - follow_through: release 이후 ~ 영상 끝
    """
    df = df.copy()
    n = len(df)
    df["phase"] = "windup"

    # 결측 보간 (감지 실패 프레임은 앞뒤로 채움 — 분류 안정성을 위해)
    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    for c in coord_cols:
        df[c] = df[c].interpolate(limit_direction="both")

    # 스무딩
    df["wrist_x_s"] = df["throw_wrist_x"].rolling(3, center=True, min_periods=1).mean()
    df["wrist_y_s"] = df["throw_wrist_y"].rolling(3, center=True, min_periods=1).mean()
    df["knee_y_s"]  = df["lead_knee_y"].rolling(3, center=True, min_periods=1).mean()
    df["ankle_y_s"] = df["front_ankle_y"].rolling(3, center=True, min_periods=1).mean()

    # 손목 속도
    vx = df["wrist_x_s"].diff()
    vy = df["wrist_y_s"].diff()
    df["wrist_speed"] = np.sqrt(vx**2 + vy**2)
    df["wrist_speed_s"] = df["wrist_speed"].rolling(3, center=True, min_periods=1).mean()

    if df["wrist_speed_s"].isna().all():
        return df  # 분류 불가 — 전부 windup으로 둔 채 반환

    # ── 1) release 중심: 손목 속도 최대 지점 ──
    # 단순히 속도만 보면 와인드업 중 팔을 휘적이는 움직임도 잡힐 수 있다.
    # 실제 투구의 release/acceleration 구간에서는 투구손이 어깨 높이 이상으로
    # 올라가 있다는 물리적 특징이 있으므로, 이 조건을 필터로 추가한다.
    # 화면 좌표는 아래로 갈수록 y가 커지므로, "손목이 어깨보다 위"는 wrist_y < shoulder_y.
    df["throw_shoulder_y_s"] = df["throw_shoulder_y"].rolling(3, center=True, min_periods=1).mean()

    arm_raised_mask = df["wrist_y_s"] < df["throw_shoulder_y_s"]

    # throw_wrist_vis가 낮은 구간은 신뢰도가 떨어지므로 가중치를 낮춤
    reliable_mask = df.get("throw_wrist_vis", pd.Series(1, index=df.index)) > 0.4

    # release 후보 = 신뢰도 충족 + 팔이 어깨보다 위로 올라간 프레임만
    candidate_mask = reliable_mask & arm_raised_mask
    speed_for_peak = df["wrist_speed_s"].where(candidate_mask, 0)

    if speed_for_peak.max() == 0:
        # 팔이 어깨 위로 올라가는 프레임이 한 번도 없으면(=투구 동작 자체가
        # 안 잡혔거나 감지 품질이 나쁘면) 조건을 완화해서 속도 기준으로만 재시도
        speed_for_peak = df["wrist_speed_s"].where(reliable_mask, 0)

    release_center = int(speed_for_peak.idxmax())

    # ── 2) 무릎 최고점 (windup → stride 경계) ──
    # 무릎 y가 가장 작은(화면에서 가장 위) 지점, release 이전 구간에서만 탐색
    pre_release = df.loc[:release_center]
    if pre_release["knee_y_s"].notna().any():
        knee_peak_idx = int(pre_release["knee_y_s"].idxmin())
    else:
        knee_peak_idx = 0

    # ── 3) 착지 시점 (stride → cocking 경계) ──
    # 이전 버전의 문제: knee_peak_idx ~ release_center 전체 구간에서 "발목이
    # 안 움직이는 첫 지점"을 찾다 보니, 그냥 다리를 벌리고 정지해 있는
    # 준비 자세도 조건을 만족해 착지로 오인되는 경우가 있었다.
    #
    # 실제 투구에서 착지는 release 직전 짧은 시간 안에 일어나므로,
    # 탐색 범위를 "release_center 직전 구간"으로 좁히고,
    # 착지 이후 무릎이 다시 굽기 시작하는(굴곡) 변곡점까지 함께 확인해
    # 진짜 착지 순간(다리가 가장 쭉 뻗은 찰나)을 찾는다.
    search_start = max(knee_peak_idx, release_center - int((release_center - knee_peak_idx) * 0.9))
    after_knee = df.loc[search_start:release_center].copy()

    if len(after_knee) > 3:
        ankle_vel = after_knee["ankle_y_s"].diff().abs()
        knee_vel  = after_knee["knee_y_s"].diff()  # 양수=무릎이 다시 내려감(굽음), 음수=계속 펴짐

        y_thr = after_knee["ankle_y_s"].quantile(0.6)
        vel_thr = ankle_vel.quantile(0.5)

        # 후보: 발목이 충분히 내려와 있고(착지 높이) + 발목 속도가 낮고(멈춤)
        #       + 그 직후 무릎이 다시 굽기 시작하는 지점(체중 이동 시작)
        candidates = after_knee[(after_knee["ankle_y_s"] >= y_thr) & (ankle_vel < vel_thr)]

        landing_idx = None
        for cand_idx in candidates.index:
            # 후보 시점 이후 5프레임 안에 무릎이 굽기 시작하는지(knee_vel이 양수로 전환) 확인
            window_end = min(cand_idx + 5, after_knee.index[-1])
            future_knee_vel = knee_vel.loc[cand_idx:window_end]
            if (future_knee_vel > 0).any():
                landing_idx = int(cand_idx)
                break

        if landing_idx is None:
            # 무릎 굴곡 조건을 만족하는 후보가 없으면, 가장 release에 가까운
            # 정지 후보를 사용 (release 직전이 착지일 가능성이 가장 높음)
            landing_idx = int(candidates.index[-1]) if not candidates.empty else search_start
    else:
        landing_idx = search_start

    # ── 4) 가속 시작점 (cocking → acceleration 경계) ──
    # 착지 이후 일정 프레임은 무조건 cocking으로 보장한 뒤(코킹 동작 자체가
    # 손목을 살짝 움직이므로 절대 속도 임계값만으로는 착지 직후 바로 넘어버릴 수 있음),
    # 그 이후 구간에서 속도가 "지속적으로 증가하는" 첫 지점을 가속 시작으로 본다.
    min_cocking_frames = max(2, int((release_center - landing_idx) * 0.15))
    cocking_floor_idx = min(landing_idx + min_cocking_frames, release_center)

    after_floor = df.loc[cocking_floor_idx:release_center]
    speed_threshold = df["wrist_speed_s"].max() * 0.25

    accel_start_idx = cocking_floor_idx
    if len(after_floor) >= 3:
        speeds = after_floor["wrist_speed_s"].values
        idxs = after_floor.index.values
        # 속도가 threshold를 넘고, 그 다음 프레임에서도 떨어지지 않는(증가 추세) 첫 지점
        for k in range(len(speeds) - 1):
            if speeds[k] >= speed_threshold and speeds[k + 1] >= speeds[k]:
                accel_start_idx = int(idxs[k])
                break
        else:
            # 증가 추세 조건을 만족하는 지점이 없으면 threshold만 넘은 첫 지점 사용
            over = after_floor[after_floor["wrist_speed_s"] >= speed_threshold]
            accel_start_idx = int(over.index[0]) if not over.empty else cocking_floor_idx

    # ── 5) release 구간 폭 ──
    # release 중심 기준 ±2프레임을 release로 분류 (속도 피크 부근의 짧은 창)
    release_window = 2
    release_start = max(0, release_center - release_window)
    release_end   = min(n - 1, release_center + release_window)

    # ── 라벨 적용 ──
    df.loc[0:knee_peak_idx, "phase"] = "windup"
    df.loc[knee_peak_idx:landing_idx, "phase"] = "stride"
    df.loc[landing_idx:accel_start_idx, "phase"] = "cocking"
    df.loc[accel_start_idx:release_start, "phase"] = "acceleration"
    df.loc[release_start:release_end, "phase"] = "release"
    df.loc[release_end:n, "phase"] = "follow_through"

    df.attrs["release_center"] = release_center
    df.attrs["knee_peak_idx"] = knee_peak_idx
    df.attrs["landing_idx"] = landing_idx
    df.attrs["accel_start_idx"] = accel_start_idx

    return df


# ──────────────────────────────────────────
# HSS 계산 (release 단계 프레임에서만)
# ──────────────────────────────────────────
def measure_hss_at_release(df: pd.DataFrame):
    """
    phase == 'release'로 분류된 프레임들에서만 HSS를 계산하고,
    그 중 정상 범위 안의 최댓값을 반환한다.

    이중 안전장치: classify_phases 단계에서 이미 "손목이 어깨보다 위"
    조건으로 release_center를 걸렀지만, release_window(±2프레임) 안의
    가장자리 프레임은 그 조건을 벗어날 수 있으므로 여기서도 같은 조건을
    한 번 더 확인해 진짜 팔이 올라간 프레임만 후보로 남긴다.

    반환값: (hss_value, frame_idx) 또는 (None, None)
    """
    release_df = df[df["phase"] == "release"].copy()
    if release_df.empty:
        return None, None

    hss_list = []
    for idx, row in release_df.iterrows():
        if pd.isna(row.get("l_hip_x")) or pd.isna(row.get("r_shoulder_x")):
            continue

        # 투구손이 어깨보다 위에 있는지 재확인 (화면 좌표는 위로 갈수록 y가 작음)
        wrist_y = row.get("throw_wrist_y")
        shoulder_y = row.get("throw_shoulder_y")
        if pd.notna(wrist_y) and pd.notna(shoulder_y) and wrist_y >= shoulder_y:
            continue  # 팔이 어깨보다 아래 = 던지는 동작이 아님, 후보 제외

        l_hip = np.array([row["l_hip_x"], row["l_hip_y"]])
        r_hip = np.array([row["r_hip_x"], row["r_hip_y"]])
        l_sh  = np.array([row["l_shoulder_x"], row["l_shoulder_y"]])
        r_sh  = np.array([row["r_shoulder_x"], row["r_shoulder_y"]])

        hip_angle = axis_angle(l_hip, r_hip)
        sh_angle  = axis_angle(l_sh, r_sh)
        hss = hip_angle - sh_angle
        if hss > 180:
            hss -= 360
        elif hss < -180:
            hss += 360

        hss_list.append((idx, hss))

    if not hss_list:
        return None, None

    hss_series = pd.Series({i: v for i, v in hss_list})
    valid = hss_series[(hss_series >= HSS_MIN) & (hss_series <= HSS_MAX)]

    if valid.empty:
        # release 구간 안에서도 비정상치만 있으면 측정 실패로 처리
        return None, None

    best_idx = int(valid.idxmax())
    return float(valid.max()), int(df.loc[best_idx, "frame"])


# ──────────────────────────────────────────
# 보폭 계산 (착지 직전/직후, stride→cocking 경계 프레임에서)
# ──────────────────────────────────────────
def measure_stride_at_landing(df: pd.DataFrame, player_height_m: float):
    """
    "투구손 손목이 어깨보다 위에 있는" 프레임들만 후보로 삼아,
    그 중에서 보폭(앞발-뒷발 발목 거리)이 가장 큰 시점을 찾는다.

    이렇게 바꾼 이유: 기존에는 classify_phases가 찾은 landing_idx
    한 프레임에만 의존했는데, 정지된 준비 자세(다리를 넓게 벌리고
    가만히 서 있는 상태)도 착지로 오인되는 경우가 있었다.
    "손이 어깨 위로 올라가 있다"는 조건은 실제로 투구 동작이 진행 중
    (코킹~릴리스 부근)이라는 강한 물리적 증거이므로, 이 조건을 만족하는
    프레임에서만 보폭을 측정하면 정지 자세를 원천적으로 배제할 수 있다.
    (화면 좌표는 위로 갈수록 y가 작아지므로 "손목이 어깨보다 위" = wrist_y < shoulder_y)
    """
    if "throw_wrist_y" not in df.columns or "throw_shoulder_y" not in df.columns:
        return None, None

    arm_raised = df["throw_wrist_y"] < df["throw_shoulder_y"]
    candidates = df[arm_raised].copy()

    if candidates.empty:
        return None, None  # 손이 어깨 위로 올라간 프레임이 한 번도 없음

    shoulder_w_median = df["shoulder_w_px"].median()
    if pd.isna(shoulder_w_median) or shoulder_w_median <= 0:
        return None, None
    scale = 0.45 / shoulder_w_median

    best_stride_m = None
    best_idx = None

    for idx, row in candidates.iterrows():
        if pd.isna(row.get("front_ankle_x")) or pd.isna(row.get("back_ankle_x")):
            continue

        f_xy = np.array([row["front_ankle_x"], row["front_ankle_y"]])
        b_xy = np.array([row["back_ankle_x"], row["back_ankle_y"]])
        stride_px = dist(f_xy, b_xy)
        stride_m = stride_px * scale
        stride_ratio = stride_m / player_height_m

        # 비정상 범위는 후보에서 제외
        if not (STRIDE_RATIO_MIN <= stride_ratio <= STRIDE_RATIO_MAX):
            continue

        if best_stride_m is None or stride_m > best_stride_m:
            best_stride_m = stride_m
            best_idx = idx

    if best_stride_m is None:
        return None, None

    return float(best_stride_m), int(df.loc[best_idx, "frame"])


# ──────────────────────────────────────────
# 신장 기반 스케일 계산 (어깨너비 가정보다 더 정확)
# ──────────────────────────────────────────
def estimate_height_based_scale(df: pd.DataFrame, player_height_m: float) -> float | None:
    """
    기존에는 "어깨너비 = 0.45m"라는 고정 가정으로 픽셀→미터 스케일을 잡았다.
    사람마다 체형이 달라 이 가정은 부정확할 수 있으므로, 본인이 입력한
    실제 신장을 기준으로 스케일을 다시 잡는다.

    방법: 영상에서 코(nose)부터 발목(ankle) 중간점까지의 픽셀 거리를
    "머리부터 발끝까지의 근사 신장"으로 보고, 이 거리가 실제 player_height_m에
    해당한다고 보정한다. 여러 프레임에서 측정해 중앙값을 사용해 노이즈를 줄인다.

    반환값: 1px당 실제 미터 거리 (scale), 측정 불가 시 None
    """
    required = ["nose_x", "nose_y", "l_ankle_x", "l_ankle_y", "r_ankle_x", "r_ankle_y"]
    if not all(c in df.columns for c in required):
        return None

    valid = df.dropna(subset=required)
    if valid.empty:
        return None

    nose = valid[["nose_x", "nose_y"]].values
    ankle_mid_x = (valid["l_ankle_x"] + valid["r_ankle_x"]) / 2
    ankle_mid_y = (valid["l_ankle_y"] + valid["r_ankle_y"]) / 2
    ankle_mid = np.stack([ankle_mid_x.values, ankle_mid_y.values], axis=1)

    px_heights = np.linalg.norm(nose - ankle_mid, axis=1)
    # 너무 작거나(사람이 화면 구석에 작게 잡힌 프레임) 큰 이상치는 중앙값으로 완화
    px_height_median = np.median(px_heights)

    if px_height_median <= 0:
        return None

    # 코~발목 거리는 정수리~발끝 실제 신장보다 살짝 짧으므로(코가 정수리보다
    # 아래에 있음) 약 1.05배 보정해 머리 위쪽 여유를 반영한다.
    HEAD_TOP_CORRECTION = 1.05
    scale = (player_height_m / HEAD_TOP_CORRECTION) / px_height_median
    return float(scale)


# ──────────────────────────────────────────
# Release Extension (투구판 기준 릴리스 거리)
# ──────────────────────────────────────────
def measure_release_extension(df: pd.DataFrame, player_height_m: float,
                                player_wingspan_m: float, max_hss_frame: int = None):
    """
    릴리스 시점에 투구판(투구를 시작한 뒷발 기준 위치)에서 얼마나 앞으로
    나가서 공을 놓는지를 계산한다. MLB의 "Extension" 지표와 동일한 개념.

    계산 방법:
        1) 신장 기반 스케일로 픽셀→미터 변환 (어깨너비 가정보다 정확)
        2) "손목이 어깨보다 위에 있는" 프레임들(=코킹~릴리스 구간 후보) 중,
           손목이 가장 앞으로(투구 방향) 나간 시점을 릴리스 시점으로 본다.
        3) 투구판 기준점 = 그 시점 기준 뒷발(back_ankle) 위치
           (투구 동작 중 뒷발은 투구판 근처에 거의 고정되어 있음)
        4) 익스텐션 = 투구판 위치 ~ 릴리스 손목 위치까지의 수평 거리
        5) 윙스팬/신장 비율로 팔 길이 보정:
           윙스팬이 신장보다 큰 사람은(대부분의 사람이 그렇듯) 같은 자세에서도
           실제 팔이 더 길어 익스텐션이 더 크게 나오는 경향이 있다.
           순수 픽셀 측정값에 (윙스팬/신장) 비율을 보정 계수로 살짝 반영해
           팔 길이 차이를 추정값에 녹여낸다.

    반환값: (extension_m, frame_idx) 또는 (None, None)
    """
    if "throw_wrist_x" not in df.columns or "throw_shoulder_y" not in df.columns:
        return None, None

    scale = estimate_height_based_scale(df, player_height_m)
    if scale is None:
        # 신장 기반 스케일을 못 구하면 기존 어깨너비 가정으로 대체
        shoulder_w_median = df["shoulder_w_px"].median()
        if pd.isna(shoulder_w_median) or shoulder_w_median <= 0:
            return None, None
        scale = 0.45 / shoulder_w_median

    arm_raised = df["throw_wrist_y"] < df["throw_shoulder_y"]
    candidates = df[arm_raised].copy()
    if candidates.empty:
        return None, None

    # 투구 진행 방향 추정: back_ankle -> front_ankle 방향이 "앞으로"
    # x좌표가 증가하는 방향인지 감소하는 방향인지를 영상 전체에서 판단
    valid_ankles = df.dropna(subset=["front_ankle_x", "back_ankle_x"])
    if valid_ankles.empty:
        return None, None
    direction_sign = np.sign(
        (valid_ankles["front_ankle_x"] - valid_ankles["back_ankle_x"]).median()
    )
    if direction_sign == 0:
        direction_sign = 1

    # 손목이 투구 진행 방향으로 가장 멀리 나간 시점 = 릴리스 시점
    candidates["wrist_forward"] = candidates["throw_wrist_x"] * direction_sign
    release_idx = candidates["wrist_forward"].idxmax()
    release_row = df.loc[release_idx]

    if pd.isna(release_row.get("back_ankle_x")) or pd.isna(release_row.get("throw_wrist_x")):
        return None, None

    # 투구판 기준점 = 해당 시점의 뒷발 위치
    rubber_x = release_row["back_ankle_x"]
    wrist_x = release_row["throw_wrist_x"]

    extension_px = abs(wrist_x - rubber_x)
    extension_m = extension_px * scale

    # 윙스팬/신장 비율로 팔 길이 보정
    wingspan_ratio = player_wingspan_m / player_height_m
    extension_m_corrected = extension_m * (0.5 + 0.5 * wingspan_ratio)
    # (보정 계수를 0.5~1.0 가중으로 완만하게 적용해 과보정을 방지)

    # 상식적 범위 검증: 익스텐션은 보통 신장의 0.5~1.3배를 넘기 어려움
    if not (0.3 * player_height_m <= extension_m_corrected <= 1.3 * player_height_m):
        return None, None

    return float(extension_m_corrected), int(release_row["frame"])


# ──────────────────────────────────────────
# 시각화: 단계 타임라인
# ──────────────────────────────────────────
def plot_phase_timeline(df: pd.DataFrame, save_path: str, title: str):
    """
    6단계 구간을 색상 막대로 표시하고, 각 구간마다 정확한 시작~끝 시간과
    길이(초)를 함께 표기한다. 구간 폭이 좁아 글자가 안 들어가는 경우
    (특히 release처럼 순간적인 구간)는 막대 위쪽으로 라벨을 빼서 화살표로 가리킨다.
    """
    fig, ax = plt.subplots(figsize=(13, 3.6))

    t = df["time_sec"].values

    # 각 단계의 연속 구간(시작, 끝) 찾기 — phase가 바뀌는 지점마다 구간을 끊는다.
    # (한 단계가 영상 안에서 여러 번 끊겨 나타나는 경우는 없다고 가정하지만,
    #  혹시 있더라도 안전하게 모든 연속 구간을 각각 표시한다.)
    phase_series = df["phase"].values
    segments = []  # (phase_name, start_time, end_time)
    seg_start_idx = 0
    for i in range(1, len(phase_series) + 1):
        if i == len(phase_series) or phase_series[i] != phase_series[seg_start_idx]:
            segments.append((
                phase_series[seg_start_idx],
                t[seg_start_idx],
                t[i - 1],
            ))
            seg_start_idx = i

    # 막대 그리기
    for phase_name, t_start, t_end in segments:
        ax.axvspan(t_start, t_end, color=PHASE_COLORS_HEX.get(phase_name, "#999"),
                   alpha=0.85, ymin=0.1, ymax=0.9)

    # 전체 타임라인 길이 기준으로 "라벨이 막대 안에 들어갈 수 있는 최소 폭" 판단
    # (이전에 0.05였는데, cocking/acceleration처럼 0.15~0.2초 구간에서
    #  텍스트가 옆 구간과 겹치는 문제가 있어 기준을 더 넉넉하게 올림)
    total_span = t[-1] - t[0] if len(t) > 1 else 1
    min_width_for_inline_label = total_span * 0.12

    label_y_inline = 0.5      # 막대 안쪽 라벨 높이
    outside_label_levels = [1.05, 1.65, 1.05]  # 바깥 라벨 높이를 3단계로 순환시켜
                                                  # 좁은 구간이 연달아 나와도 안 겹치게 함
    outside_label_count = 0

    for phase_name, t_start, t_end in segments:
        duration = t_end - t_start
        mid = (t_start + t_end) / 2
        # 이름 / 시작-끝(길이) — 2줄로 압축해 박스 높이 안에 안전하게 들어가게 함
        label_text = f"{phase_name}\n{t_start:.2f}–{t_end:.2f}s ({duration:.2f}s)"

        if duration >= min_width_for_inline_label:
            # 막대 안에 라벨이 들어갈 만큼 넓으면 막대 중앙에 표시
            ax.text(mid, label_y_inline, label_text, ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold",
                    linespacing=1.6)
        else:
            # 좁은 구간(코킹, 가속, 릴리스 등)은 막대 위로 라벨을 빼고 화살표로 연결.
            # 바깥 라벨은 3줄로 표시해도 공간 제약이 없으므로 줄을 나눠 가독성을 높인다.
            label_text_outside = f"{phase_name}\n{t_start:.2f}–{t_end:.2f}s\n({duration:.2f}s)"
            y_pos = outside_label_levels[outside_label_count % len(outside_label_levels)]
            outside_label_count += 1
            ax.annotate(
                label_text_outside,
                xy=(mid, 0.9), xytext=(mid, y_pos),
                ha="center", va="bottom", fontsize=7.5, color="#333",
                linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color="#888", lw=1),
            )

    ax.set_ylim(0, 2.6)
    ax.set_yticks([])
    ax.set_xlabel("시간 (초)")
    ax.set_title(title, fontsize=12)

    # 범례는 색상-단계명 매칭용으로 하단에 별도 유지
    handles = [mpatches.Patch(color=PHASE_COLORS_HEX[p], label=p) for p in PHASES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=6, fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  단계 타임라인 저장: {save_path}")


# ──────────────────────────────────────────
# 검증용 오버레이 이미지
# ──────────────────────────────────────────
def save_overlay_image(video_path, model_path, target_frame, label, save_path, mode="hss",
                        num_poses=1, pitcher_select="smallest", throws_right=True):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  (검증 이미지 실패: 프레임 {target_frame} 읽기 불가)")
        return

    scale_factor = 3 if w > 1500 else 1
    proc = cv2.resize(frame, (w // scale_factor, h // scale_factor)) if scale_factor > 1 else frame
    ph, pw = proc.shape[:2]

    with create_landmarker(model_path, RunningMode.IMAGE, num_poses=num_poses) as lmkr:
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = lmkr.detect(mp_img)

    lms = select_pitcher_pose(result.pose_landmarks, mode=pitcher_select) if result.pose_landmarks else None
    if lms is None:
        print(f"  (검증 이미지: 프레임 {target_frame} 포즈 미감지)")
        return

    def pt(idx):
        return (int(lms[idx].x * pw), int(lms[idx].y * ph))

    if mode == "stride":
        cv2.line(proc, pt(IDX["l_ankle"]), pt(IDX["r_ankle"]), (0, 165, 255), 4)
    elif mode == "extension":
        throw_wrist_idx = IDX["r_wrist"] if throws_right else IDX["l_wrist"]
        back_ankle_idx = IDX["r_ankle"] if throws_right else IDX["l_ankle"]
        cv2.line(proc, pt(back_ankle_idx), pt(throw_wrist_idx), (0, 220, 255), 4)
        cv2.circle(proc, pt(back_ankle_idx), 10, (0, 140, 255), 2)  # 투구판 위치 강조
    else:
        cv2.line(proc, pt(IDX["l_hip"]), pt(IDX["r_hip"]), (0, 165, 255), 3)
        cv2.line(proc, pt(IDX["l_shoulder"]), pt(IDX["r_shoulder"]), (180, 60, 220), 3)

    for idx in IDX.values():
        if lms[idx].visibility > 0.3:
            cv2.circle(proc, pt(idx), 6, (0, 200, 120), -1)

    cv2.putText(proc, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 150), 2)
    cv2.putText(proc, f"Frame: {target_frame}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    cv2.imwrite(str(save_path), proc)
    print(f"  검증 이미지 저장: {save_path}")


# ──────────────────────────────────────────
# 영상 자동 탐색 & 선택
# ──────────────────────────────────────────
def select_video_interactively(search_dir: str, extensions: list) -> str:
    """
    search_dir 안에서 영상 파일들을 찾아 번호 목록으로 보여주고,
    사람이 번호를 입력해 고르게 한다. 새 영상을 찍을 때마다 코드 안의
    파일명을 고칠 필요 없이, 그냥 이 폴더에 영상을 옮겨놓기만 하면 된다.

    목록에 없는 파일을 쓰고 싶으면 0번을 선택해 경로를 직접 입력할 수 있다.
    """
    search_path = Path(search_dir)
    videos = sorted([
        p for p in search_path.iterdir()
        if p.is_file() and p.suffix in extensions
    ])

    if not videos:
        print(f"'{search_dir}' 폴더에서 영상 파일을 찾지 못했습니다.")
        manual = input("분석할 영상의 파일명(또는 경로)을 직접 입력하세요: ").strip()
        return manual

    print("\n분석할 영상을 선택하세요:")
    for i, v in enumerate(videos, start=1):
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {v.name}  ({size_mb:.1f}MB)")
    print(f"  [0] 목록에 없는 다른 영상 — 파일명 직접 입력")

    while True:
        choice = input(f"번호 입력 (1~{len(videos)}, 0): ").strip()
        if choice == "0":
            manual = input("분석할 영상의 파일명(또는 경로)을 입력하세요: ").strip()
            return manual
        if choice.isdigit() and 1 <= int(choice) <= len(videos):
            return str(videos[int(choice) - 1])
        print("잘못된 입력입니다. 다시 시도하세요.")


def make_output_dir(video_path: str) -> Path:
    """
    영상 파일명 기준으로 결과 저장 폴더를 만든다.
    예: IMG_9225_2.mov -> results/IMG_9225_2/
    여러 영상을 분석해도 결과가 서로 덮어쓰이지 않고 영상별로 따로 쌓인다.
    """
    video_stem = Path(video_path).stem
    out_dir = Path("results") / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_PATH = select_video_interactively(VIDEO_SEARCH_DIR, VIDEO_EXTENSIONS)
    OUT_DIR = make_output_dir(VIDEO_PATH)
    print(f"\n선택된 영상: {VIDEO_PATH}")
    print(f"결과 저장 위치: {OUT_DIR}/\n")

    print(f"[{VIDEO_PATH}] 관절 추출 중...")
    df, fps = extract_all_landmarks(
        VIDEO_PATH, MODEL_PATH, THROWS_RIGHT,
        num_poses=NUM_POSES, pitcher_select=PITCHER_SELECT_MODE
    )
    print("동작 단계 분류 중...")
    df = classify_phases(df)
    plot_phase_timeline(df, str(OUT_DIR / "phase_timeline.png"), "투구 동작 단계 (측면 영상)")

    stride_m, stride_frame = measure_stride_at_landing(df, PLAYER_HEIGHT_M)
    extension_m, extension_frame = measure_release_extension(df, PLAYER_HEIGHT_M, PLAYER_WINGSPAN_M)
    max_hss, max_hss_frame = measure_hss_at_release(df)

    print("\n" + "="*40)
    print("  투구 메커니즘 분석 결과 (릴리스 시점 기준)")
    print("="*40)

    if stride_m is None:
        print("Stride length: 측정 실패 (착지 시점 또는 정상 범위 내 값 없음)")
    else:
        print(f"Stride length: {stride_m:.2f}m  (착지 프레임 기준)")
        save_overlay_image(VIDEO_PATH, MODEL_PATH, stride_frame,
                            f"Stride: {stride_m:.2f}m", str(OUT_DIR / "stride_check_frame.png"), mode="stride",
                            num_poses=NUM_POSES, pitcher_select=PITCHER_SELECT_MODE,
                            throws_right=THROWS_RIGHT)

    if extension_m is None:
        print("Release extension: 측정 실패 (release 구간 내 정상 값 없음)")
    else:
        print(f"Release extension: {extension_m:.2f}m  (투구판 기준 릴리스 거리, 신장/윙스팬 보정)")
        save_overlay_image(VIDEO_PATH, MODEL_PATH, extension_frame,
                            f"Extension: {extension_m:.2f}m", str(OUT_DIR / "extension_check_frame.png"), mode="extension",
                            num_poses=NUM_POSES, pitcher_select=PITCHER_SELECT_MODE,
                            throws_right=THROWS_RIGHT)

    if max_hss is None:
        print("Max hip-shoulder separation: 측정 실패 (release 구간 내 정상 값 없음)")
    else:
        print(f"Max hip-shoulder separation: {max_hss:.1f}°  (release 단계 프레임 기준)")
        save_overlay_image(VIDEO_PATH, MODEL_PATH, max_hss_frame,
                            f"HSS: {max_hss:.1f} deg (release)", str(OUT_DIR / "hss_check_frame.png"), mode="hss",
                            num_poses=NUM_POSES, pitcher_select=PITCHER_SELECT_MODE,
                            throws_right=THROWS_RIGHT)

    print("="*40)
    print(f"\n결과 폴더: {OUT_DIR}/")
    print("단계 타임라인 이미지(phase_timeline.png)를 확인하면")
    print("release/stride 구간이 어디로 잡혔는지 볼 수 있습니다.")
