"""
static.py  —  静态手势识别窗口
摄像头采集由 camera_worker.CameraWorker 负责；
识别逻辑由本文件内的 RecognitionThread 处理。
"""

import sys
import os
import json
import time
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from collections import Counter

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSlider, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSizePolicy, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from camera import CameraWorker

# ══════════ 全局配置 ══════════
ROOT          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(ROOT, "model", "class.onnx")
JSON_PATH     = os.path.join(ROOT, "model", "class.json")
CAMERA_ID     = 0
CONF_THRESH   = 0.75
BUFFER_SIZE   = 20
BLANK_TIMEOUT = 15

LET2NUM = {
    "I": "1", "V": "2", "OK": "3", "B": "4", "U": "5", "Y": "6",
    "Q": "7", "R": "8", "J": "9", "D": "10", "X": "10", "H": "10",
    "O": "0", "C": "0",
}
NUM2LET = {"5": "U", "4": "B"}


# ══════════ 识别线程（纯逻辑，不采集摄像头）══════════
class RecognitionThread(QThread):
    """
    接收来自 CameraWorker 的 BGR 帧，完成手势识别后
    发出 frame_ready(annotated_frame, pred_dict_or_None, seq_str)。
    """
    frame_ready = pyqtSignal(np.ndarray, object, str)

    def __init__(self, model_path, json_path):
        super().__init__()
        self.recognizing      = False
        self.mode_index       = 0
        self.conf_thresh      = CONF_THRESH

        self.history_buffer   = []
        self.final_sequence   = []
        self.last_name        = None
        self.blank_counter    = 0
        self.continue_counter = 0
        self.last_clear_time  = 0
        self.ema_coords       = None
        self.alpha            = 0.7
        self.all_conf         = 0.0

        # 当前待处理帧（由主线程写入）
        self._pending_frame   = None
        self._run_flag        = True

        with open(json_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            model_complexity=1,
        )
        self.mp_draw = mp.solutions.drawing_utils

    # ── 公开接口 ─────────────────────────────────────────────────
    def push_frame(self, frame: np.ndarray):
        """CameraWorker.frame_ready 连接到这里。"""
        self._pending_frame = frame.copy()

    def clear_sequence(self):
        self._clear_data()

    def stop(self):
        self._run_flag = False
        self.wait()

    # ── 内部辅助 ─────────────────────────────────────────────────
    def _get_priors(self, lm):
        def d(i, j): return np.linalg.norm(lm[i] - lm[j])
        return np.array([d(4, 8), d(8, 0), d(12, 0), d(8, 12), d(4, 0)],
                        np.float32)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _update_sequence(self, name, conf, raw_name):
        if self.last_name in ("5", "U") and raw_name == "D":
            self._clear_data()
            return
        if time.time() - self.last_clear_time < 1.0:
            return
        if name != self.last_name:
            self.final_sequence.append(name)
            self.all_conf         += conf
            self.last_name         = name
            self.continue_counter  = 0
        else:
            self.continue_counter += 1
            if self.continue_counter > 20:
                self.final_sequence.append(name)
                self.all_conf         += conf
                self.continue_counter  = 0

    def _clear_data(self):
        self.final_sequence   = []
        self.history_buffer   = []
        self.all_conf         = 0.0
        self.continue_counter = 0
        self.last_name        = None
        self.ema_coords       = None
        self.last_clear_time  = time.time()

    # ── 线程主循环 ───────────────────────────────────────────────
    def run(self):
        prev_time = time.time()

        while self._run_flag:
            frame = self._pending_frame
            if frame is None:
                self.msleep(10)
                continue
            self._pending_frame = None

            # 统一格式
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 200, 60), 2)

            pred = None

            if self.recognizing:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)

                if results.multi_hand_landmarks and results.multi_handedness:
                    self.blank_counter = 0
                    temp = []
                    for hand_lm, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):
                        label      = handedness.classification[0].label
                        draw_color = (255, 80, 80) if label == "Left" else (80, 80, 255)
                        hand_spec  = self.mp_draw.DrawingSpec(
                            color=draw_color, thickness=2, circle_radius=2)
                        self.mp_draw.draw_landmarks(
                            frame, hand_lm,
                            self.mp_hands.HAND_CONNECTIONS, hand_spec
                        )

                        coords = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                            np.float32,
                        )
                        if self.ema_coords is None:
                            self.ema_coords = coords
                        else:
                            self.ema_coords = (
                                self.alpha * coords
                                + (1 - self.alpha) * self.ema_coords
                            )
                        ec = self.ema_coords.copy()
                        ec -= ec[0]
                        mx = np.max(np.abs(ec))
                        if mx:
                            ec /= mx

                        priors     = self._get_priors(ec)
                        feat_input = np.concatenate(
                            [ec.flatten(), priors]
                        ).reshape(1, -1).astype(np.float32)
                        inp_name = self.session.get_inputs()[0].name
                        out      = self.session.run(None, {inp_name: feat_input})
                        probs    = self._softmax(out[0][0])
                        idx      = int(np.argmax(probs))
                        conf     = float(probs[idx])
                        temp.append({
                            "name": self.class_names[idx],
                            "conf": conf,
                            "raw_name": self.class_names[idx],
                        })

                    if temp:
                        best     = max(temp, key=lambda x: x["conf"])
                        name     = best["name"]
                        conf     = best["conf"]
                        raw_name = best["raw_name"]
                        if conf >= self.conf_thresh:
                            self.history_buffer.append(name)
                            if len(self.history_buffer) > BUFFER_SIZE:
                                self.history_buffer.pop(0)
                            name = Counter(self.history_buffer).most_common(1)[0][0]
                            display = (
                                NUM2LET.get(name, name)
                                if self.mode_index == 0
                                else LET2NUM.get(name, name)
                            )
                            self._update_sequence(display, conf, raw_name)
                            pred = {"name": display, "conf": conf}
                        else:
                            display = (
                                NUM2LET.get(name, name)
                                if self.mode_index == 0
                                else LET2NUM.get(name, name)
                            )
                            pred = {"name": display, "conf": conf}
                else:
                    self.blank_counter += 1
                    pred = None
                    if self.blank_counter > BLANK_TIMEOUT:
                        self.last_name        = None
                        self.continue_counter = 0

            seq_str = "".join(self.final_sequence)
            self.frame_ready.emit(frame, pred, seq_str)
            time.sleep(1 / 30)

        self.hands.close()


# ══════════ 静态识别主窗口 ══════════
class StaticWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("孤立词识别系统 — 静态识别")
        self.resize(800, 400)

        self._mode_index      = 0
        self._cam_worker      = None   # CameraWorker
        self._rec_thread      = None   # RecognitionThread
        self._current_source  = CAMERA_ID

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # ── 左：视频区 ──────────────────────────────────────────
        video_container = QWidget()
        video_container.setStyleSheet(
            "background-color: #e8e8e8; border-radius: 10px;"
        )
        video_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        vl = QVBoxLayout(video_container)
        vl.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel("摄像头启动中...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("color: #999; font-size: 16px;")
        vl.addWidget(self.image_label)
        main_layout.addWidget(video_container, stretch=3)

        # ── 右：控制面板 ────────────────────────────────────────
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignmentFlag.AlignTop)
        right.setSpacing(0)

        # 识别结果
        res_title = QLabel("识别结果")
        res_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_title.setStyleSheet(
            "color: #555; font-size: 13px; font-weight: bold; padding-top: 4px;"
        )
        right.addWidget(res_title)
        right.addSpacing(4)

        self.result_display = QLabel("—")
        self.result_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_display.setWordWrap(True)
        self.result_display.setMinimumHeight(90)
        self._reset_result()
        right.addWidget(self.result_display)
        right.addSpacing(8)

        # 识别阈值滑块（与 dynamic 风格一致）
        self.thresh_tag = QLabel(f"识别阈值: {CONF_THRESH:.2f}")
        self.thresh_tag.setStyleSheet(
            "color: #0077aa; font-size: 12px; font-weight: bold;"
        )
        right.addWidget(self.thresh_tag)

        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(int(CONF_THRESH * 100))
        self.slider_thresh.setStyleSheet("""
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
        right.addWidget(self.slider_thresh)

        def update_thresh(val):
            new_conf = val / 100.0
            self.thresh_tag.setText(f"识别阈值: {new_conf:.2f}")
            if self._rec_thread:
                self._rec_thread.conf_thresh = new_conf

        self.slider_thresh.valueChanged.connect(update_thresh)
        right.addSpacing(8)

        # 识别模式
        mode_label = QLabel("识别模式")
        mode_label.setStyleSheet(
            "color: #555; font-size: 12px; font-weight: bold;"
        )
        right.addWidget(mode_label)
        right.addSpacing(4)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.btn_letter = QPushButton("字母")
        self.btn_number = QPushButton("数字")
        for btn in (self.btn_letter, self.btn_number):
            btn.setMinimumHeight(36)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(True)
        self.btn_letter.setChecked(True)
        self.btn_letter.clicked.connect(lambda: self._set_mode(0))
        self.btn_number.clicked.connect(lambda: self._set_mode(1))
        self._update_mode_style()
        mode_row.addWidget(self.btn_letter)
        mode_row.addWidget(self.btn_number)
        right.addLayout(mode_row)
        right.addSpacing(8)

        # 拼接序列
        seq_title = QLabel("拼接序列")
        seq_title.setStyleSheet(
            "color: #555; font-size: 12px; font-weight: bold;"
        )
        right.addWidget(seq_title)
        right.addSpacing(4)
        self.seq_display = QLabel("")
        self.seq_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.seq_display.setWordWrap(True)
        self.seq_display.setMinimumHeight(45)
        self.seq_display.setStyleSheet("""
            QLabel {
                background-color: #eaf6ff; color: #0077aa;
                border: 1px solid #b3d9f0; border-radius: 8px;
                font-size: 20px; font-weight: bold;
                font-family: 'Consolas', 'Courier New', monospace; padding: 6px;
            }
        """)
        right.addWidget(self.seq_display)
        right.addSpacing(8)

        # 操作按钮
        self.btn_live = QPushButton("实时识别")
        self.btn_file   = QPushButton("选择文件")
        self.btn_clear  = QPushButton("清除序列")

        for btn, c, h, p in [
            (self.btn_live, "#4CAF50", "#43A047", "#388E3C"),
            (self.btn_file,  "#2196F3", "#1E88E5", "#1565C0"),
            (self.btn_clear, "#FF9800", "#FB8C00", "#E65100"),
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
        self.btn_clear.clicked.connect(self._clear)

        right.addStretch()
        for btn in (self.btn_live, self.btn_file, self.btn_clear):
            right.addWidget(btn)
            right.addSpacing(6)

        main_layout.addLayout(right, stretch=1)

        self._launch_source(CAMERA_ID)

    # ── 源切换 ───────────────────────────────────────────────────
    def _launch_source(self, source):
        self._stop_threads()
        self._current_source = source
        self._reset_result()
        self.seq_display.setText("")

        # 识别线程
        self._rec_thread = RecognitionThread(MODEL_PATH, JSON_PATH)
        self._rec_thread.mode_index  = self._mode_index
        self._rec_thread.conf_thresh = self.slider_thresh.value() / 100.0
        self._rec_thread.recognizing = True
        self._rec_thread.frame_ready.connect(self._on_frame)
        self._rec_thread.start()

        # 摄像头线程
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

    # ── UI 事件 ──────────────────────────────────────────────────
    def _open_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", "All Files (*.*)"
        )
        if f:
            self._launch_source(f)

    def _switch_to_live(self):
        self._launch_source(CAMERA_ID)
        self._reset_result()
        print("已切换至实时模式并自动开启识别")

    def _clear(self):
        current_text = self.seq_display.text().strip()
        if current_text and self._rec_thread:
            source = self._current_source
            if isinstance(source, str):
                ext = source.lower()
                in_mode_str = (
                    "图片"
                    if ext.endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp"))
                    else "视频"
                )
            else:
                in_mode_str = "实时"
            mode_name = "字母" if self._mode_index == 0 else "数字"
            avg_conf = 0.0
            seq = self._rec_thread.final_sequence
            if seq:
                avg_conf = self._rec_thread.all_conf / len(seq)

            from db_manager import DBManager
            db = DBManager()
            db.insert_record(
                result=current_text,
                conf=round(float(avg_conf), 2),
                in_mode=in_mode_str,
                rec_mode=mode_name,
            )
            print(f"数据已成功保存到 history.db: {current_text}")

        if self._rec_thread:
            self._rec_thread.clear_sequence()
        self.seq_display.setText("")

    def _set_mode(self, idx):
        self._mode_index = idx
        self.btn_letter.setChecked(idx == 0)
        self.btn_number.setChecked(idx == 1)
        self._update_mode_style()
        if self._rec_thread:
            self._rec_thread.mode_index = idx

    def _update_mode_style(self):
        active   = ("background-color: #4CAF50; color: white;"
                    " border-radius: 6px; font-size: 13px; font-weight: bold;"
                    " border: none;")
        inactive = ("background-color: #e0e0e0; color: #666;"
                    " border-radius: 6px; font-size: 13px; border: none;")
        self.btn_letter.setStyleSheet(
            f"QPushButton {{ {active if self._mode_index == 0 else inactive} }}"
        )
        self.btn_number.setStyleSheet(
            f"QPushButton {{ {active if self._mode_index == 1 else inactive} }}"
        )

    # ── 帧回调 ───────────────────────────────────────────────────
    def _on_frame(self, cv_img, pred, seq_str):
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
        if pred:
            conf  = pred["conf"]
            name  = pred["name"]
            thresh = (
                self._rec_thread.conf_thresh if self._rec_thread else CONF_THRESH
            )
            if conf >= thresh:
                color  = "#2e7d32"
                border = "#4CAF50"
                bg     = "#f1f8f1"
            else:
                color  = "#e65100"
                border = "#FF9800"
                bg     = "#fff8f0"
            self.result_display.setText(f"{name}\n{conf * 100:.1f}%")
            self.result_display.setStyleSheet(f"""
                QLabel {{
                    background-color: {bg}; color: {color};
                    border: 2px solid {border}; border-radius: 10px;
                    font-size: 28px; font-weight: bold;
                    font-family: 'Microsoft YaHei', sans-serif; padding: 10px;
                }}
            """)
        else:
            self._reset_result()
        self.seq_display.setText(seq_str)

    def _reset_result(self):
        self.result_display.setText("—")
        self.result_display.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5; color: #aaa;
                border: 2px solid #e0e0e0; border-radius: 10px;
                font-size: 22px; font-weight: bold;
                font-family: 'Microsoft YaHei', sans-serif; padding: 10px;
            }
        """)

    def closeEvent(self, event):
        self._stop_threads()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QWidget     { background-color: #f5f5f5; color: #212121; }
    """)
    win = StaticWindow()
    win.show()
    sys.exit(app.exec())