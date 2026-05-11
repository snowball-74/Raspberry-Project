import sys
import os
import subprocess
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from history_ui import HistoryWindow

ROOT = os.path.dirname(os.path.abspath(__file__))


class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("孤立词识别系统")
        self.setFixedSize(600, 400)
        self.setStyleSheet("background-color: #f5f5f5;")

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(30, 20, 30, 20)

        lbl = QLabel("^o^")
        lbl.setFont(QFont("Microsoft YaHei", 50))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #999999; margin-bottom: 5px;")
        layout.addWidget(lbl)

        buttons = [
            ("静态识别", "#4CAF50", "#43A047", "#388E3C", self._launch_static),
            ("动态识别", "#2196F3", "#1E88E5", "#1565C0", self._launch_dynamic),
            ("历史记录", "#FF9800", "#FB8C00", "#E65100", self._launch_history),
        ]

        for text, color, hover, pressed, func in buttons:
            btn = QPushButton(text)
            btn.setFixedWidth(300)
            btn.setMinimumHeight(80)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color}; color: white;
                    border-radius: 8px; font-size: 30px; font-weight: bold;
                    border: none;
                }}
                QPushButton:hover   {{ background-color: {hover}; }}
                QPushButton:pressed {{ background-color: {pressed}; }}
            """)
            btn.clicked.connect(func)
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _launch_static(self):
        subprocess.Popen([sys.executable, os.path.join(ROOT, "static.py")])

    def _launch_dynamic(self):
        subprocess.Popen([sys.executable, os.path.join(ROOT, "dynamic.py")])

    def _launch_history(self):
        from db_manager import DBManager
        self.db = DBManager()
        self.hi = HistoryWindow(self.db)
        self.hi.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QWidget     { background-color: #f5f5f5; color: #212121; }
    """)
    win = Launcher()
    win.show()
    sys.exit(app.exec())