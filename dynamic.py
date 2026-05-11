"""
dynamic.py  —  动态手势识别窗口
摄像头采集由 camera_worker.CameraWorker 负责；
识别逻辑保留在本文件内的 RecognitionThread（原 VideoThread 重命名）。
"""

import sys, os
import cv2
import mediapipe as mp
import numpy as np
import collections
import time
import json
import onnxruntime as ort

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSlider, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSizePolicy, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from camera import CameraWorker

# ══════════ 1. 全局配置参数 ══════════
ROOT     = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN  = 20
FEAT_DIM = 172
POSE_DIM = 40
VOTE_WINDOW  = 8
POSE_FRAMES  = 3

DISPLAY_THRESH = 0.65
MONITOR_THRESH = 0.20

CAMERA_ID = 0

KP_WRIST  = 0
KP_THUMB  = 4
KP_INDEX  = 8
KP_MIDDLE = 12
KP_RING   = 16
KP_PINKY  = 20
FIVE_TIPS = [KP_THUMB, KP_INDEX, KP_MIDDLE, KP_RING, KP_PINKY]
FOUR_TIPS = [KP_INDEX, KP_MIDDLE, KP_RING, KP_PINKY]
FINGER_JOINTS = [
    (1, 2,  4), (5, 6,  8), (9, 10, 12), (13, 14, 16), (17, 18, 20),
]

C_BLACK  = (0,   0,   0)
C_WHITE  = (255, 255, 255)
C_GREEN  = (40,  200, 60)
C_GRAY   = (130, 130, 130)
C_YELLOW = (0,   210, 230)
TIP_COLORS = [
    (30, 160, 255), (255, 170, 30), (200, 80, 255), (50, 50, 240), (50, 220, 100),
]


# ══════════ 2. 特征提取函数（与原版完全一致）══════════

def lm_to_numpy(landmarks, num_points=21):
    if not landmarks:
        return np.zeros((num_points, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                    dtype=np.float32)

def get_pose_point(pose, idx):
    if not pose or idx >= len(pose.landmark):
        return np.zeros(3, dtype=np.float32)
    lm = pose.landmark[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def palm_center(lm_21x3):
    if np.all(lm_21x3 == 0):
        return np.zeros(3, dtype=np.float32)
    return (lm_21x3[0] + lm_21x3[9]) * 0.5

def robust_assign_hands_live(pw_l, pw_r, rh_raw, lh_raw):
    zero21   = np.zeros((21, 3), dtype=np.float32)
    rh_blank = np.all(rh_raw == 0)
    lh_blank = np.all(lh_raw == 0)
    if rh_blank and lh_blank:
        return zero21.copy(), zero21.copy()
    pw_r_2d = pw_r[:2]
    pw_l_2d = pw_l[:2]
    if not rh_blank and not lh_blank:
        d_rh_to_r = np.linalg.norm(rh_raw[0][:2] - pw_r_2d)
        d_lh_to_r = np.linalg.norm(lh_raw[0][:2] - pw_r_2d)
        return (rh_raw, lh_raw) if d_rh_to_r <= d_lh_to_r else (lh_raw, rh_raw)
    hand   = rh_raw if not rh_blank else lh_raw
    d_to_r = np.linalg.norm(hand[0][:2] - pw_r_2d)
    d_to_l = np.linalg.norm(hand[0][:2] - pw_l_2d)
    return (hand, zero21.copy()) if d_to_r <= d_to_l else (zero21.copy(), hand)

def safe_vec(src, dst):
    if np.all(src == 0) or np.all(dst == 0):
        return np.zeros(3, dtype=np.float32)
    return (dst - src).astype(np.float32)

def finger_bend_angles(lm_21x3):
    if np.all(lm_21x3 == 0):
        return np.zeros(5, dtype=np.float32)
    angles = np.zeros(5, dtype=np.float32)
    for k, (a, b, c) in enumerate(FINGER_JOINTS):
        ba = lm_21x3[a] - lm_21x3[b]
        bc = lm_21x3[c] - lm_21x3[b]
        n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if n_ba < 1e-6 or n_bc < 1e-6:
            angles[k] = 0.0
        else:
            angles[k] = np.arccos(
                np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
            )
    return angles

def hand_features_86(lm, lm_prev, nose, opp_pose_wrist, opp_palm,
                     opp_shoulder, same_shoulder, opp_elbow, l_sho, r_sho):
    out = np.zeros(86, dtype=np.float32)
    hand_blank = np.all(lm == 0)
    sho_center = (l_sho + r_sho) * 0.5
    if not hand_blank:
        r1   = lm[KP_THUMB]
        palm = palm_center(lm)
        for k, tgt in enumerate([lm[KP_INDEX], lm[KP_MIDDLE], lm[KP_RING],
                                  lm[KP_PINKY], palm]):
            out[k*3 : k*3+3] = safe_vec(r1, tgt)
        for k, tid in enumerate(FOUR_TIPS):
            out[15 + k*3 : 15 + k*3+3] = safe_vec(palm, lm[tid])
        for k, tgt in enumerate([nose, opp_pose_wrist, opp_palm,
                                  opp_shoulder, same_shoulder, opp_elbow]):
            out[27 + k*3 : 27 + k*3+3] = safe_vec(palm, tgt)
        r3 = lm[KP_WRIST]
        for k, tgt in enumerate([nose, opp_pose_wrist, opp_elbow,
                                  same_shoulder, opp_shoulder]):
            out[45 + k*3 : 45 + k*3+3] = safe_vec(r3, tgt)
    out[60:65] = finger_bend_angles(lm)
    if not hand_blank:
        palm       = palm_center(lm)
        prev_blank = (lm_prev is None) or np.all(lm_prev == 0)
        if not prev_blank:
            prev_palm = palm_center(lm_prev)
            for k, tid in enumerate(FIVE_TIPS):
                out[65 + k*3 : 65 + k*3+3] = (
                    lm[tid] - palm - (lm_prev[tid] - prev_palm)
                ).astype(np.float32)
            out[80:83] = (
                (palm - sho_center) - (prev_palm - sho_center)
            ).astype(np.float32)
            r3, prev_r3 = lm[KP_WRIST], lm_prev[KP_WRIST]
            out[83:86] = (
                (r3 - sho_center) - (prev_r3 - sho_center)
            ).astype(np.float32)
    return out

def extract_frame_features(results, prev_rh, prev_lh, w, h):
    if not results.pose_landmarks:
        return None, None, None, None
    pose    = results.pose_landmarks
    nose    = get_pose_point(pose, 0)
    l_sho   = get_pose_point(pose, 11)
    r_sho   = get_pose_point(pose, 12)
    l_elbow = get_pose_point(pose, 13)
    r_elbow = get_pose_point(pose, 14)
    pw_l    = get_pose_point(pose, 15)
    pw_r    = get_pose_point(pose, 16)
    rh_raw  = lm_to_numpy(results.right_hand_landmarks, 21)
    lh_raw  = lm_to_numpy(results.left_hand_landmarks,  21)
    true_rh, true_lh = robust_assign_hands_live(pw_l, pw_r, rh_raw, lh_raw)
    r_palm  = palm_center(true_rh)
    l_palm  = palm_center(true_lh)
    feat    = np.zeros(172, dtype=np.float32)
    feat[0:86]   = hand_features_86(
        true_rh, prev_rh, nose, pw_l, l_palm, l_sho, r_sho, l_elbow, l_sho, r_sho)
    feat[86:172] = hand_features_86(
        true_lh, prev_lh, nose, pw_r, r_palm, r_sho, l_sho, r_elbow, l_sho, r_sho)
    pix_rh, pix_lh = {}, {}
    if not np.all(true_rh == 0):
        for kid in [KP_WRIST] + FIVE_TIPS:
            pix_rh[kid] = (int(true_rh[kid][0] * w), int(true_rh[kid][1] * h))
        pix_rh[KP_WRIST] = (int(r_palm[0] * w), int(r_palm[1] * h))
    if not np.all(true_lh == 0):
        for kid in [KP_WRIST] + FIVE_TIPS:
            pix_lh[kid] = (int(true_lh[kid][0] * w), int(true_lh[kid][1] * h))
        pix_lh[KP_WRIST] = (int(l_palm[0] * w), int(l_palm[1] * h))
    return feat, true_rh, true_lh, (pix_rh, pix_lh)

def preprocess_all(raw_buf, norm_params):
    arr  = np.array(raw_buf, dtype=np.float32)[-SEQ_LEN:]
    if len(arr) < SEQ_LEN:
        arr = np.pad(arr, ((0, SEQ_LEN - len(arr)), (0, 0)), 'constant')
    boost = np.array(norm_params['vel_boost'], np.float32)
    mean  = np.array(norm_params['seq_mean'],  np.float32).flatten()
    std   = np.array(norm_params['seq_std'],   np.float32).flatten()
    arr   = (arr * boost[None, :] - mean) / std

    def get_pose(sub_buf, is_end):
        sub_arr  = np.array(sub_buf, np.float32)
        rh_tips  = sub_arr[:, 15:27].mean(axis=0)
        rh_thumb = sub_arr[:, 12:15].mean(axis=0)
        rh_bend  = sub_arr[:, 60:65].mean(axis=0)
        lh_tips  = sub_arr[:, 86+15 : 86+27].mean(axis=0)
        lh_thumb = sub_arr[:, 86+12 : 86+15].mean(axis=0)
        lh_bend  = sub_arr[:, 86+60 : 86+65].mean(axis=0)
        p  = np.concatenate([rh_tips, rh_thumb, rh_bend, lh_tips, lh_thumb, lh_bend])
        pm = np.array(norm_params['p_mean']).flatten()
        ps = np.array(norm_params['p_std']).flatten()
        if is_end:
            return ((p - pm[POSE_DIM:]) / ps[POSE_DIM:]).astype(np.float32)
        else:
            return ((p - pm[:POSE_DIM]) / ps[:POSE_DIM]).astype(np.float32)

    return (arr.astype(np.float32),
            get_pose(raw_buf[:POSE_FRAMES], False),
            get_pose(raw_buf[-POSE_FRAMES:], True))


# ══════════ 3. 识别线程（接收 CameraWorker 帧）══════════
class RecognitionThread(QThread):
    """
    从 CameraWorker 接收 BGR 帧，完成 holistic 识别和动态手势推理。
    信号：
        annotated_frame_signal(np.ndarray)  — 带绘制信息的 BGR 帧
        monitor_signal(str, float)          — 监控日志
        result_signal(str, float)           — 高置信度最终结果
    """
    annotated_frame_signal = pyqtSignal(np.ndarray)
    monitor_signal         = pyqtSignal(str, float)
    result_signal          = pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag    = True
        self.recognizing  = True
        self.display_thresh = DISPLAY_THRESH
        self.is_file=False

        self._pending_frame = None   # 由主线程写入

        try:
            class_path = os.path.join(ROOT, "model", "class_names.json")
            norm_path  = os.path.join(ROOT, "model", "norm_params.json")
            model_path = os.path.join(ROOT, "model", "dynamic_model.onnx")
            with open(class_path, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
            with open(norm_path, "r", encoding="utf-8") as f:
                self.norm_params = json.load(f)
            self.ort_session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider'])
            print(f"[ONNX] 加载成功. 类别数: {len(self.class_names)}")
        except Exception as e:
            print(f"[ERROR] 资源加载失败: {e}")
            self.ort_session = None

    # ── 公开接口 ────────────────────────────────────────────────
    def push_frame(self, frame: np.ndarray):
        """CameraWorker.frame_ready 连接到这里。"""
        self._pending_frame = frame.copy()

    def stop(self):
        self._run_flag = False
        self.wait()

    # ── 线程主循环 ──────────────────────────────────────────────
    def run(self):
        mp_holistic = mp.solutions.holistic
        mp_drawing  = mp.solutions.drawing_utils
        holistic    = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        raw_buf = []
        wrist_trail_r, wrist_trail_l = [], []
        tip_trail_r, tip_trail_l     = [], []
        vote_buf         = collections.deque(maxlen=VOTE_WINDOW)
        prev_rh, prev_lh = None, None
        prev_time = time.time()

        def draw_trails(frame, wrist_trail, tip_trail):
            for i in range(1, len(wrist_trail)):
                p1, p2 = wrist_trail[i-1], wrist_trail[i]
                if p1 and p2:
                    cv2.line(frame, p1, p2, C_WHITE,
                             max(1, int(3 * (i / len(wrist_trail)))))
            for fi, kid in enumerate(FIVE_TIPS):
                color = TIP_COLORS[fi % len(TIP_COLORS)]
                for i in range(1, len(tip_trail)):
                    t1, t2 = tip_trail[i-1], tip_trail[i]
                    if t1 and t2 and kid in t1 and kid in t2:
                        cv2.line(frame, t1[kid], t2[kid], color,
                                 max(1, int(2 * (i / len(tip_trail)))))

        while self._run_flag:
            frame = self._pending_frame
            if frame is None:
                self.msleep(10)
                continue
            self._pending_frame = None

            h, w = frame.shape[:2]
            now = time.time()
            fps = 1.0 / max(now-prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS {fps:.0f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40,200,60),2)
            if not self.is_file:
                frame = cv2.flip(frame,1)            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                res = holistic.process(img_rgb)
            except Exception as e:
                print(f"[WARN] holistic 异常: {e}")
                res = None

            if res and self.recognizing and self.ort_session:
                if res.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, res.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS)
                if res.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, res.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS)

                feat, true_rh, true_lh, pix = extract_frame_features(
                    res, prev_rh, prev_lh, w, h)
                rh_blank = np.all(true_rh == 0) if true_rh is not None else True
                lh_blank = np.all(true_lh == 0) if true_lh is not None else True

                if feat is not None and (not rh_blank or not lh_blank):
                    prev_rh, prev_lh = true_rh, true_lh
                    raw_buf.append(feat)
                    pix_rh, pix_lh = pix
                    wrist_trail_r.append(pix_rh.get(KP_WRIST) if pix_rh else None)
                    tip_trail_r.append(pix_rh if pix_rh else None)
                    wrist_trail_l.append(pix_lh.get(KP_WRIST) if pix_lh else None)
                    tip_trail_l.append(pix_lh if pix_lh else None)
                    if len(raw_buf) > SEQ_LEN:
                        raw_buf.pop(0)
                        wrist_trail_r.pop(0); tip_trail_r.pop(0)
                        wrist_trail_l.pop(0); tip_trail_l.pop(0)

                    draw_trails(frame, wrist_trail_r, tip_trail_r)
                    draw_trails(frame, wrist_trail_l, tip_trail_l)

                    if len(raw_buf) >= 10:
                        try:
                            s_np, sp_np, ep_np = preprocess_all(
                                raw_buf, self.norm_params)
                            inputs = {
                                self.ort_session.get_inputs()[0].name: s_np[np.newaxis, ...],
                                self.ort_session.get_inputs()[1].name: sp_np[np.newaxis, ...],
                                self.ort_session.get_inputs()[2].name: ep_np[np.newaxis, ...],
                            }
                            ort_outs   = self.ort_session.run(None, inputs)
                            logits     = ort_outs[0]
                            exp_logits = np.exp(
                                logits - np.max(logits, axis=1, keepdims=True))
                            probs      = (exp_logits / np.sum(
                                exp_logits, axis=1, keepdims=True))[0]
                            pred       = int(np.argmax(probs))
                            pred_name  = self.class_names[pred]
                            pred_conf  = float(probs[pred])
                            if pred_conf > MONITOR_THRESH and pred_name != "void":
                                self.monitor_signal.emit(pred_name, pred_conf)
                            vote_buf.append((pred, pred_conf))
                            top_v    = collections.Counter(
                                v for v, _ in vote_buf).most_common(1)[0][0]
                            top_conf = float(np.mean(
                                [c for v, c in vote_buf if v == top_v]))
                            top_name = self.class_names[top_v]
                            if top_name != "void":
                                self.result_signal.emit(top_name, top_conf)
                        except Exception as e:
                            print(f"[WARN] 推理异常: {e}")
                else:
                    prev_rh, prev_lh = None, None
            else:
                prev_rh, prev_lh = None, None

            # 缓冲进度条
            if self.recognizing:
                bx, by, bw_bar, bh_bar = 10, h - 20, 160, 8
                cv2.rectangle(frame, (bx, by), (bx + bw_bar, by + bh_bar),
                              C_GRAY, 1)
                if raw_buf:
                    fill_w = int((len(raw_buf) / SEQ_LEN) * bw_bar)
                    fill_c = C_GREEN if len(raw_buf) == SEQ_LEN else C_YELLOW
                    cv2.rectangle(frame, (bx, by),
                                  (bx + fill_w, by + bh_bar), fill_c, -1)
            out_frame = cv2.flip(frame, 1) if not self.is_file else frame
            self.annotated_frame_signal.emit(out_frame)
            time.sleep(1 / 30)

        holistic.close()


# ══════════ 4. 主界面 ══════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("孤立词识别系统 — 动态识别")
        self.resize(800, 400)

        self._cam_worker      = None
        self._rec_thread      = None
        self.last_saved_label = None
        self.is_file          = False
        self._result_hold     = False
        
        from PyQt6.QtCore import QTimer
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(1500)
        self._hold_timer.timeout.connect(self._on_hold_expired)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # ── 左：视频区 ──────────────────────────────────────────
        self.video_container = QWidget()
        self.video_container.setStyleSheet(
            "background-color: #e8e8e8; border-radius: 10px;"
        )
        self.video_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        video_layout = QVBoxLayout(self.video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel("摄像头启动中...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("color: #999; font-size: 16px;")
        video_layout.addWidget(self.image_label)
        main_layout.addWidget(self.video_container, stretch=3)

        # ── 右：控制面板 ────────────────────────────────────────
        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_panel.setSpacing(0)

        result_title = QLabel("识别结果")
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_title.setStyleSheet(
            "color: #555; font-size: 13px; font-weight: bold; padding-top: 4px;"
        )
        right_panel.addWidget(result_title)
        right_panel.addSpacing(4)

        self.result_display = QLabel("—")
        self.result_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_display.setWordWrap(True)
        self.result_display.setMinimumHeight(90)
        self._reset_result_display()
        right_panel.addWidget(self.result_display)
        right_panel.addSpacing(8)

        # 显示阈值滑块
        self.thresh_tag = QLabel(f"识别阈值: {DISPLAY_THRESH:.2f}")
        self.thresh_tag.setStyleSheet(
            "color: #0077aa; font-size: 13px; font-weight: bold;"
        )
        right_panel.addWidget(self.thresh_tag)

        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(30, 90)
        self.slider_conf.setValue(int(DISPLAY_THRESH * 100))
        self.slider_conf.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #d0e8f5; height: 6px; border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #2196F3; height: 6px; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196F3; width: 16px; height: 16px;
                margin: -5px 0; border-radius: 8px;
                border: 2px solid #ffffff;
            }
            QSlider::handle:horizontal:hover {
                background: #1565C0;
            }
        """)
        right_panel.addWidget(self.slider_conf)

        def on_conf_change(v):
            val = v / 100.0
            self.thresh_tag.setText(f"显示阈值: {val:.2f}")
            if self._rec_thread:
                self._rec_thread.display_thresh = val

        self.slider_conf.valueChanged.connect(on_conf_change)
        right_panel.addSpacing(8)

        # 实时识别按钮
        self.btn_live = QPushButton("实时识别")
        self.btn_file = QPushButton("选择文件")
        
        for btn, c, h, p in [
            (self.btn_live, "#4CAF50", "#43A047", "#388E3C"),
            (self.btn_file,  "#2196F3", "#1E88E5", "#1565C0"),
        ]:
            btn.setMinimumHeight(44)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    font-weight: bold; background-color: {c};
                    color: white; border-radius: 8px; font-size: 14px;
                    border: none;
                }}
                QPushButton:hover   {{ background-color: {h}; }}
                QPushButton:pressed {{ background-color: {p}; }}
                QPushButton:disabled {{ background-color: #cccccc; color: #888888; }}
            """)
        self.btn_live.clicked.connect(self._switch_to_live)
        self.btn_file.clicked.connect(self._open_file)

        # 实时监控日志
        mon_label = QLabel("实时监控日志")
        mon_label.setStyleSheet(
            "color: #0077aa; font-size: 12px; font-weight: bold; padding: 2px 0;"
        )
        right_panel.addWidget(mon_label)
        right_panel.addSpacing(6)

        self.monitor_console = QTextEdit()
        self.monitor_console.setReadOnly(True)
        self.monitor_console.setMinimumHeight(150)
        self.monitor_console.setStyleSheet("""
            QTextEdit {
                background-color: #f0faff; color: #005577;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px; border: 1px solid #b3d9f0;
                border-radius: 6px; padding: 4px;
            }
        """)
        right_panel.addWidget(self.monitor_console, stretch=1)
        main_layout.addLayout(right_panel, stretch=1)

        right_panel.addSpacing(8)
        right_panel.addWidget(self.btn_live)
        right_panel.addSpacing(8)
        right_panel.addWidget(self.btn_file)
        right_panel.addSpacing(8)

        self._launch_source(CAMERA_ID)

    # ── 源切换 ───────────────────────────────────────────────────
    def _launch_source(self, source):
        self._stop_threads()
        self.is_file = isinstance(source, str)

        self._rec_thread = RecognitionThread()
        self._rec_thread.is_file = self.is_file
        self._rec_thread.annotated_frame_signal.connect(self.update_image)
        self._rec_thread.monitor_signal.connect(self.update_monitor_log)
        self._rec_thread.result_signal.connect(self.update_result_display)
        self._rec_thread.recognizing = True
        self._rec_thread.start()

        self._cam_worker = CameraWorker()
        self._cam_worker.set_source(source)
        self._cam_worker.frame_ready.connect(self._rec_thread.push_frame)
        self._cam_worker.start_camera()

    def _stop_threads(self):
        if self._cam_worker:
            self._cam_worker.stop_camera()
            self._cam_worker = None
        if self._rec_thread:
            self._rec_thread.stop()
            self._rec_thread = None

    def _switch_to_live(self):
        self._stop_threads()
        QThread.msleep(100)
        self._launch_source(CAMERA_ID)
        self.last_saved_label = None
        self._reset_result_display()
        print("已切换至实时模式并自动开启识别")

    def _open_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "Videos (*.mp4 *.MP4 *.MTS *.mkv *.wmv *.avi)"
        )
        if f:
            self._reset_result_display()
            self._launch_source(f)

    # ── UI 更新 ──────────────────────────────────────────────────
    def update_result_display(self, label: str, conf: float):
        thresh = self._rec_thread.display_thresh if self._rec_thread else DISPLAY_THRESH
        if label and conf >= thresh:
            self._result_hold = True
            self._hold_timer.start()
            self.result_display.setText(f"{label}\n{conf * 100:.1f}%")
            self.result_display.setStyleSheet("""
                QLabel {
                    background-color: #f1f8f1; color: #2e7d32;
                    border: 2px solid #4CAF50; border-radius: 10px;
                    font-size: 26px; font-weight: bold;
                    font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
                    padding: 10px;
                }
            """)
            if label != self.last_saved_label:
                in_mode = "视频" if self.is_file else "实时"
                from db_manager import DBManager
                db = DBManager()
                db.insert_record(
                    result=label,
                    conf=round(float(conf), 2),
                    in_mode=in_mode,
                    rec_mode="动态",
                )
                self.last_saved_label = label
                print(f"检测到高置信度结果，已存库: {label} ({conf:.2f})")
        else:
            if not self._result_hold:
                self._reset_result_display()
                self.last_saved_label = None
                
    def _on_hold_expired(self):
        self._result_hold = False
        self._reset_result_display()
        self.last_saved_label = None
        
    def _reset_result_display(self):
        self.result_display.setText("—")
        self.result_display.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5; color: #aaa;
                border: 2px solid #e0e0e0; border-radius: 10px;
                font-size: 22px; font-weight: bold;
                font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
                padding: 10px;
            }
        """)

    def update_monitor_log(self, label: str, conf: float):
        t = time.strftime("%H:%M:%S")
        self.monitor_console.append(f"[{t}]  {label}  {conf:.2f}")
        sb = self.monitor_console.verticalScrollBar()
        sb.setValue(sb.maximum())
        if self.monitor_console.document().blockCount() > 200:
            self.monitor_console.clear()

    def update_image(self, cv_img: np.ndarray):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(
            QPixmap.fromImage(qi).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def closeEvent(self, event):
        self._stop_threads()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QWidget     { background-color: #f5f5f5; color: #212121; }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())