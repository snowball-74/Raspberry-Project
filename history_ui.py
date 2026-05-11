from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QCheckBox, QLineEdit, QLabel, QComboBox,
    QApplication, QMessageBox, QMenu
)
from PyQt6.QtCore import Qt


class HistoryWindow(QDialog):
    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        self.setWindowTitle("Recognition History")
        self.resize(800, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # ── 筛选栏 ────────────────────────────────────────────────
        filter_layout = QHBoxLayout()

        # 搜索框
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("搜索识别结果...")
        self.edit_search.setFixedWidth(150)
        self.edit_search.textChanged.connect(self.load_data)

        # 识别大模式筛选（静态 / 动态）
        self.combo_algo_mode = QComboBox()
        self.combo_algo_mode.addItems(["所有算法模式", "静态识别", "动态识别"])
        self.combo_algo_mode.currentIndexChanged.connect(self.load_data)

        # 识别子模式筛选（字母 / 数字 — 仅静态有子模式）
        self.combo_rec_mode = QComboBox()
        self.combo_rec_mode.addItems(["所有子模式", "字母", "数字"])
        self.combo_rec_mode.currentIndexChanged.connect(self.load_data)

        # 输入模式筛选
        self.combo_in_mode = QComboBox()
        self.combo_in_mode.addItems(["所有输入模式", "实时", "视频", "图片"])
        self.combo_in_mode.currentIndexChanged.connect(self.load_data)

        # 置信度筛选
        self.combo_conf_range = QComboBox()
        self.combo_conf_range.addItems(["所有置信度", "> 0.8", "> 0.6", "> 0.3"])
        self.combo_conf_range.currentIndexChanged.connect(self.load_data)

        filter_layout.addWidget(QLabel("搜索:"))
        filter_layout.addWidget(self.edit_search)
        filter_layout.addSpacing(6)
        filter_layout.addWidget(self.combo_conf_range)
        filter_layout.addWidget(self.combo_in_mode)
        filter_layout.addWidget(self.combo_algo_mode)
        filter_layout.addWidget(self.combo_rec_mode)
        filter_layout.addStretch()

        # 全选 / 批量删除
        self.btn_select_all   = QPushButton("全选")
        self.btn_batch_delete = QPushButton("批量删除")
        self.btn_batch_delete.setStyleSheet(
            "background-color: #f44336; color: white;")
        self.btn_select_all.clicked.connect(self.toggle_select_all)
        self.btn_batch_delete.clicked.connect(self.delete_selected)
        filter_layout.addWidget(self.btn_select_all)
        filter_layout.addWidget(self.btn_batch_delete)
        layout.addLayout(filter_layout)

        # ── 表格 ──────────────────────────────────────────────────
        # 列: [0]checkbox [1]ID(hidden) [2]识别结果 [3]置信度
        #     [4]输入模式 [5]识别模式 [6]保存时间 [7]删除按钮
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setColumnHidden(1, True)
        self.table.setHorizontalHeaderLabels(
            ["", "ID", "识别结果", "置信度", "输入模式", "识别模式", "保存时间", ""])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 24)

        for col in [3, 4]:
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(col, 70)

        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(5, 110)

        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(6, 145)

        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(7, 32)

        layout.addWidget(self.table)
        self.load_data()

    # ── 数据加载 ──────────────────────────────────────────────────

    def load_data(self):
        records = self.db.fetch_all()
        self.table.setRowCount(0)

        search_text  = self.edit_search.text().lower()
        algo_filter  = self.combo_algo_mode.currentText()   # 静态/动态/所有
        rec_filter   = self.combo_rec_mode.currentText()    # 字母/数字/所有
        in_filter    = self.combo_in_mode.currentText()
        conf_filter  = self.combo_conf_range.currentText()

        for row_data in records:
            # row_data: (id, result, confidence, input_mode, recognize_mode, time)
            res_val  = str(row_data[1]).lower()
            conf_val = float(row_data[2])
            in_val   = str(row_data[3])
            rec_val  = str(row_data[4])   # e.g. "静态-字母", "静态-数字", "动态"

            # 搜索框
            if search_text and search_text not in res_val:
                continue

            # 算法模式筛选
            if algo_filter == "静态识别" and rec_val not in ["字母", "数字"]:
                continue
            if algo_filter == "动态识别" and rec_val != "动态":
                continue

            # 子模式筛选（只对静态有意义）
            if rec_filter == "字母" and "字母" not in rec_val:
                continue
            if rec_filter == "数字" and "数字" not in rec_val:
                continue

            # 输入模式
            if in_filter != "所有输入模式" and in_filter not in in_val:
                continue

            # 置信度
            if conf_filter == "> 0.8" and conf_val < 0.8: continue
            if conf_filter == "> 0.6" and conf_val < 0.6: continue
            if conf_filter == "> 0.3" and conf_val < 0.3: continue

            # ── 渲染行 ────────────────────────────────────────────
            i = self.table.rowCount()
            self.table.insertRow(i)

            chk = QCheckBox()
            self.table.setCellWidget(i, 0, chk)

            for j, val in enumerate(row_data):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip(str(val))
                self.table.setItem(i, j + 1, item)

            btn_del = QPushButton("❌")
            btn_del.setFixedWidth(32)
            btn_del.clicked.connect(
                lambda _, r=row_data[0]: self.delete_single(r))
            self.table.setCellWidget(i, 7, btn_del)

        self.btn_select_all.setText("全选")

    # ── 选择 / 删除 ───────────────────────────────────────────────

    def toggle_select_all(self):
        rows = self.table.rowCount()
        if rows == 0:
            self.btn_select_all.setText("全选")
            return
        first_chk = self.table.cellWidget(0, 0)
        target    = not first_chk.isChecked()
        for i in range(rows):
            chk = self.table.cellWidget(i, 0)
            if chk:
                chk.setChecked(target)
        self.btn_select_all.setText("取消全选" if target else "全选")

    def delete_selected(self):
        ids = []
        for i in range(self.table.rowCount()):
            chk = self.table.cellWidget(i, 0)
            if chk and chk.isChecked():
                ids.append(self.table.item(i, 1).text())
        if ids:
            for r_id in ids:
                self.db.delete_record(r_id)
            self.load_data()

    def delete_single(self, record_id):
        self.db.delete_record(record_id)
        self.load_data()

    # ── 右键菜单 ──────────────────────────────────────────────────

    def show_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if item and item.column() == 2:
            menu = QMenu(self)
            copy_action    = menu.addAction("复制")
            preview_action = menu.addAction("预览")
            action = menu.exec(self.table.mapToGlobal(pos))
            if action == copy_action:
                QApplication.clipboard().setText(item.text())
            elif action == preview_action:
                msg = QMessageBox(self)
                msg.setWindowTitle("识别结果预览")
                msg.setText(item.text())
                copy_btn = msg.addButton("复制内容",
                                         QMessageBox.ButtonRole.ActionRole)
                msg.addButton(QMessageBox.StandardButton.Close)
                msg.exec()
                if msg.clickedButton() == copy_btn:
                    QApplication.clipboard().setText(item.text())