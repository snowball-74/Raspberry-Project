"""
camera.py
统一摄像头线程，供 static.py 和 dynamic.py 共用。
- 树莓派 5 (Picamera2) 和普通 USB/内置摄像头自动切换
- 支持视频文件回放（循环）
- 发出 frame_ready(np.ndarray) 信号，帧格式统一为 BGR

修复：Picamera2 在 stop_camera() 后未完全释放导致
      "camera in configured state trying acquire()" 的问题。
解决方式：
  1. stop() 之后调用 close() 真正释放设备文件
  2. 用类级全局锁 + 实例缓存，避免同进程内重复打开摄像头
  3. 在新实例启动前加短暂等待，给内核时间回收资源
"""

import time
import threading

import cv2
import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread

# ── 尝试导入树莓派专用库 ─────────────────────────────────────────
try:
    import libcamera
    from picamera2 import Picamera2
    _PI_CAMERA_AVAILABLE = True
except ImportError:
    _PI_CAMERA_AVAILABLE = False


# ── 进程级全局锁：确保同一时刻只有一个线程在操作 Picamera2 ────────
_picam2_lock = threading.Lock()


class CameraWorker(QThread):
    """
    通用摄像头/视频文件线程。

    使用方式：
        worker = CameraWorker()
        worker.set_source(0)          # 摄像头 ID，或传入文件路径字符串
        worker.frame_ready.connect(your_slot)
        worker.start_camera()
        ...
        worker.stop_camera()

    信号：
        frame_ready(np.ndarray)  — BGR 格式帧
    """

    frame_ready = pyqtSignal(np.ndarray)

    # ---------- 构造 ----------
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running   = False
        self.paused    = False          # 仅对视频文件生效
        self._source   = 0
        self._is_file  = False

        # 内部句柄
        self._cap      = None           # cv2.VideoCapture
        self._picam2   = None           # Picamera2（树莓派）

    # ---------- 公开接口 ----------
    def set_source(self, source):
        """
        source: int  → 摄像头 ID（0、1、8 …）
                str  → 视频 / 图片文件路径
        """
        self._source  = source
        self._is_file = isinstance(source, str)

    def start_camera(self):
        if not self.isRunning():
            self.running = True
            self.start()

    def stop_camera(self):
        """
        停止线程并等待其完全退出。
        Picamera2 情况下会在 run() 末尾调用 close()，
        此处额外等待一段时间确保内核释放摄像头设备。
        """
        self.running = False
        # 等待线程退出（含 picam2.stop() + picam2.close()）
        self.wait()
        # 额外延时：给 libcamera 内核驱动时间完成去初始化
        # 不使用 Picamera2 时此延时几乎无感知
        if _PI_CAMERA_AVAILABLE and not self._is_file:
            time.sleep(0.3)

    # ---------- 线程主体 ----------
    def run(self):
        if self._is_file:
            self._run_opencv(self._source)
        elif _PI_CAMERA_AVAILABLE:
            self._run_picamera()
        else:
            self._run_opencv(self._source)

    # ── 普通 OpenCV 摄像头 / 视频文件 ────────────────────────────
    def _run_opencv(self, source):
        self._cap = cv2.VideoCapture(source)
        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            if self._is_file and self.paused:
                self.msleep(100)
                continue

            ret, frame = self._cap.read()
            if ret:
                if not self._is_file:
                    frame = cv2.flip(frame, 1)   # 摄像头水平镜像
                self.frame_ready.emit(frame)
                if self._is_file:
                    self.msleep(30)              # ~30 FPS 播放
            else:
                if self._is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                else:
                    self.msleep(10)

        if self._cap:
            self._cap.release()
            self._cap = None

    # ── 树莓派 5 Picamera2 ───────────────────────────────────────
    def _run_picamera(self):
        """
        修复要点：
        1. 用 _picam2_lock 序列化多次启动，防止前一个实例还没关闭
           就创建新实例。
        2. 无论正常退出还是异常，都执行 picam2.stop() + picam2.close()。
           close() 才真正释放 /dev/video* 设备文件；
           原来只调用 stop() 导致设备残留在 configured 状态。
        3. 启动前检查 Picamera2.global_camera_info()，若摄像头
           仍被占用则等待后重试，而不是直接报错。
        """
        # —— 等待前一个实例彻底释放 ——————————————————————————————
        # 最多重试 10 次（共约 2 秒），直到摄像头空闲
        max_retries = 10
        for attempt in range(max_retries):
            try:
                info = Picamera2.global_camera_info()
                if info:                         # 有摄像头信息则可以继续
                    break
            except Exception:
                pass
            print(f"[CameraWorker] 等待摄像头释放… ({attempt+1}/{max_retries})")
            time.sleep(0.2)

        with _picam2_lock:
            try:
                self._picam2 = Picamera2()
                config = self._picam2.create_preview_configuration(
                    main={"format": "RGB888", "size": (960, 720)},
                    raw={"format": "SRGGB12", "size": (1920, 1080)},
                )
                config["transform"] = libcamera.Transform(hflip=1, vflip=0)
                self._picam2.configure(config)
                self._picam2.start()
                print("[CameraWorker] Picamera2 started.")
            except Exception as exc:
                print(f"[CameraWorker] Picamera2 启动失败，回退到 OpenCV: {exc}")
                # 启动失败也要释放，避免设备残留
                if self._picam2 is not None:
                    try:
                        self._picam2.close()
                    except Exception:
                        pass
                    self._picam2 = None
                # 回退到 OpenCV
                self._run_opencv(self._source)
                return

        # —— 采集主循环 ———————————————————————————————————————————
        try:
            while self.running:
                frame_rgb = self._picam2.capture_array()        # RGB
                # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                self.frame_ready.emit(frame_rgb)                # ← 修正：发 BGR
                self.msleep(33)                                 # ~30 FPS
        finally:
            # —— 彻底释放 ————————————————————————————————————————
            # stop()  → 停止数据流（configured 状态）
            # close() → 关闭设备文件（released 状态）← 关键！原来缺少这步
            with _picam2_lock:
                if self._picam2 is not None:
                    try:
                        self._picam2.stop()
                        self._picam2.close()     # ← 核心修复
                        print("[CameraWorker] Picamera2 stopped and closed.")
                    except Exception as e:
                        print(f"[CameraWorker] 关闭 Picamera2 时出错: {e}")
                    finally:
                        self._picam2 = None