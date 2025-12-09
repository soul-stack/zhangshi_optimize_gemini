# -*- coding: utf-8 -*-
"""
炉子参数维护 GUI（带累计时间 & 炉子启用开关 & 参数类型分表日志 & 产品类型信息维护）
- 支持 A/B/C 三条产线
- 每条产线维护 N2/N3/N4/N5 四个炉子：启用状态 / 温度 / 通气量 / 加盐量
- 每条产线独立维护统一“时间微调”累计值 time_total
- 管理产品类型信息（目标渗层、最短时间等），供 API 使用

日志规则：
- Excel 中按“参数类型”分 Sheet：时间调整、温度调整、通气量调整、加盐量调整
- 炉子参数三个 Sheet 的列为：日期时间、产线、炉号、炉子是否启用、调整前值、调整量、调整后值
- 时间调整 Sheet 的列为：日期时间、产线、调整前值、调整量、调整后值、总调整时间、总产线水平时间
"""

import json
import sys
import traceback
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QComboBox,
    QFileDialog, QDoubleSpinBox, QCheckBox, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtCore import Qt


# ==================== 路径：共享 DATA_ROOT ====================
def get_data_root() -> Path:
    """
    与 nitriding_service_zh_api 保持一致：
    - exe：返回 exe 的上一级目录（dist）
    - 源码：返回当前 .py 所在目录
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        return exe_dir.parent
    return Path(__file__).resolve().parent


DATA_ROOT = get_data_root()

PARAM_FILE = DATA_ROOT / "furnace_params.json"
PRODUCT_CONF_FILE = DATA_ROOT / "product_config.json"  # 产品配置文件
LOG_CONFIG_FILE = DATA_ROOT / "furnace_log_config.json"
ADJUST_LOG_XLSX = DATA_ROOT / "furnace_adjust_log.xlsx"

# ==================== 初始产品数据 (参考 Excel) ====================
INITIAL_PRODUCT_DATA = {
    "A": {"std": "7~12", "target": 9.5, "min_time": 48},
    "B": {"std": "12~20", "target": 16.0, "min_time": 80},
    "C": {"std": "12~20", "target": 16.0, "min_time": 80},
    "D": {"std": "12~20", "target": 16.0, "min_time": 80},
    "E": {"std": "8~12", "target": 10.0, "min_time": 60},
    "F": {"std": "8~12", "target": 10.0, "min_time": 63},
    "G": {"std": "6~10", "target": 8.0, "min_time": 57},
    "H": {"std": "15~18", "target": 16.5, "min_time": 111},
    "I": {"std": "15~20", "target": 17.5, "min_time": 111},
    "J": {"std": "4~7", "target": 5.5, "min_time": 48},
    "K": {"std": "6~8", "target": 7.0, "min_time": 51},
    "L": {"std": "8~12", "target": 10.0, "min_time": 69},
    "M": {"std": "12~20", "target": 16.0, "min_time": 80},
    "N": {"std": "10~14", "target": 12.0, "min_time": 84},
    "O": {"std": "12~20", "target": 16.0, "min_time": 84},
}


# ==================== 配置读写 ====================
def load_params() -> Dict[str, Any]:
    if PARAM_FILE.exists():
        try:
            with open(PARAM_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    for line in ("A", "B", "C"):
        if line not in data:
            data[line] = {"furnaces": {}, "time_adjust": 0.0, "time_total": 0.0}
        if "furnaces" not in data[line]:
            data[line]["furnaces"] = {}
        for fname in ("N2", "N3", "N4", "N5"):
            if fname not in data[line]["furnaces"]:
                data[line]["furnaces"][fname] = {
                    "enabled": True,
                    "temperature": None,
                    "air_flow": None,
                    "salt": None,
                }
            else:
                if "enabled" not in data[line]["furnaces"][fname]:
                    data[line]["furnaces"][fname]["enabled"] = True
        if "time_adjust" not in data[line]:
            data[line]["time_adjust"] = 0.0
        if "time_total" not in data[line]:
            try:
                data[line]["time_total"] = float(data[line].get("time_adjust", 0.0))
            except Exception:
                data[line]["time_total"] = 0.0

    return data


def save_params(data: Dict[str, Any]):
    PARAM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAM_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_product_config() -> Dict[str, Dict[str, Any]]:
    """加载产品配置，如果不存在则创建默认配置"""
    if not PRODUCT_CONF_FILE.exists():
        save_product_config(INITIAL_PRODUCT_DATA)
        return INITIAL_PRODUCT_DATA

    try:
        with open(PRODUCT_CONF_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 简单校验
            if not isinstance(data, dict):
                return INITIAL_PRODUCT_DATA
            return data
    except Exception:
        return INITIAL_PRODUCT_DATA


def save_product_config(data: Dict[str, Dict[str, Any]]):
    PRODUCT_CONF_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRODUCT_CONF_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_log_path() -> Path:
    if LOG_CONFIG_FILE.exists():
        try:
            with open(LOG_CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            p = Path(cfg.get("log_path", str(ADJUST_LOG_XLSX)))
            return p
        except Exception:
            return ADJUST_LOG_XLSX
    return ADJUST_LOG_XLSX


def save_log_path(p: Path):
    LOG_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"log_path": str(p)}, f, ensure_ascii=False, indent=2)


# ==================== Excel 追加工具 ====================
def append_log_rows(path: Path, df: pd.DataFrame, sheet_name: str, parent_widget: QWidget):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                old = pd.read_excel(path, sheet_name=sheet_name)
                new_df = pd.concat([old, df], ignore_index=True)
            except Exception:
                new_df = df
            with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                new_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    except PermissionError:
        QMessageBox.warning(parent_widget, "提示", f"Excel 文件已打开：\n{path}\n\n请先在 Excel 中关闭该文件后再保存。")
    except Exception as e:
        QMessageBox.warning(parent_widget, "写入 Excel 日志时出错", f"{e}")


# ==================== 产品信息维护窗口 ====================
class ProductConfigManager(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("产品信息维护")
        self.resize(700, 500)
        self.data = load_product_config()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel("说明：此处设定将直接影响 API 的渗氮时间计算。目标值即为优化的目标厚度。")
        info.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["产品类型", "渗层基准 (文本)", "目标值 (厚度 mm/um)", "最短时间 (分)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        # 按钮区
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("新增产品")
        btn_add.clicked.connect(self.add_row)
        btn_del = QPushButton("删除选中")
        btn_del.clicked.connect(self.delete_rows)
        btn_save = QPushButton("保存配置")
        btn_save.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.Bold))
        btn_save.clicked.connect(self.save_config)

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_del)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

        self.load_table_data()

    def load_table_data(self):
        self.table.setRowCount(0)
        # 按键名排序
        keys = sorted(self.data.keys())
        for k in keys:
            info = self.data[k]
            row = self.table.rowCount()
            self.table.insertRow(row)

            # 类型
            self.table.setItem(row, 0, QTableWidgetItem(k))
            # 基准
            self.table.setItem(row, 1, QTableWidgetItem(str(info.get("std", ""))))
            # 目标值
            self.table.setItem(row, 2, QTableWidgetItem(str(info.get("target", 0))))
            # 最短时间
            self.table.setItem(row, 3, QTableWidgetItem(str(info.get("min_time", 0))))

    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("New"))
        self.table.setItem(row, 1, QTableWidgetItem("10~15"))
        self.table.setItem(row, 2, QTableWidgetItem("12.5"))
        self.table.setItem(row, 3, QTableWidgetItem("60"))
        self.table.scrollToBottom()

    def delete_rows(self):
        rows = set()
        for item in self.table.selectedItems():
            rows.add(item.row())
        for r in sorted(list(rows), reverse=True):
            self.table.removeRow(r)

    def save_config(self):
        new_data = {}
        try:
            for r in range(self.table.rowCount()):
                t_item = self.table.item(r, 0)
                std_item = self.table.item(r, 1)
                tgt_item = self.table.item(r, 2)
                min_item = self.table.item(r, 3)

                if not t_item or not t_item.text().strip():
                    continue

                ptype = t_item.text().strip().upper()
                std_val = std_item.text().strip() if std_item else ""

                try:
                    target_val = float(tgt_item.text().strip())
                except:
                    QMessageBox.warning(self, "错误", f"产品 {ptype} 的目标值必须是数字")
                    return

                try:
                    min_val = float(min_item.text().strip())
                except:
                    QMessageBox.warning(self, "错误", f"产品 {ptype} 的最短时间必须是数字")
                    return

                new_data[ptype] = {
                    "std": std_val,
                    "target": target_val,
                    "min_time": min_val
                }

            self.data = new_data
            save_product_config(self.data)
            QMessageBox.information(self, "成功", "产品配置已保存！API 将立即使用新配置。")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))


# ==================== GUI 主体 ====================
class FurnaceManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("炉子参数与产品维护系统")
        self.resize(880, 520)

        # 确保初始产品配置存在
        load_product_config()

        self.params = load_params()
        self.current_line = "A"
        self.log_path = load_log_path()

        self._dirty = False
        self._loading = False

        self._build_ui()
        self._load_line("A")

    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel("炉子参数与产品维护")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # 顶部：产线选择 + 产品信息入口
        top = QHBoxLayout()
        top.addWidget(QLabel("产线："))
        self.combo_line = QComboBox()
        self.combo_line.addItems(["A", "B", "C"])
        self.combo_line.currentTextChanged.connect(self.on_line_change)
        top.addWidget(self.combo_line)

        top.addSpacing(20)
        btn_prod = QPushButton("产品信息维护")
        btn_prod.setStyleSheet("background-color: #e1f5fe; color: #01579b; font-weight: bold;")
        btn_prod.clicked.connect(self.open_product_manager)
        top.addWidget(btn_prod)

        top.addStretch(1)

        self.ed_log = QLineEdit(str(self.log_path))
        self.ed_log.setReadOnly(True)
        self.ed_log.setMinimumWidth(250)
        btn_choose_log = QPushButton("选择日志路径")
        btn_choose_log.clicked.connect(self.choose_log_path)

        top.addWidget(QLabel("调整日志："))
        top.addWidget(self.ed_log)
        top.addWidget(btn_choose_log)
        root.addLayout(top)

        # 炉子参数表格
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel("炉号"), 0, 0)
        grid.addWidget(QLabel("启用"), 0, 1)
        grid.addWidget(QLabel("温度"), 0, 2)
        grid.addWidget(QLabel("通气量"), 0, 3)
        grid.addWidget(QLabel("加盐量"), 0, 4)

        self.edits: Dict[Tuple[str, str], QLineEdit] = {}
        self.chk_enabled: Dict[str, QCheckBox] = {}

        row = 1
        for fname in ("N2", "N3", "N4", "N5"):
            grid.addWidget(QLabel(fname), row, 0)

            chk = QCheckBox()
            chk.setChecked(True)
            chk.stateChanged.connect(self.on_enabled_changed)
            self.chk_enabled[fname] = chk
            grid.addWidget(chk, row, 1)

            for col, key in enumerate(("temperature", "air_flow", "salt"), start=2):
                e = QLineEdit()
                e.setPlaceholderText("--")
                e.textEdited.connect(self.mark_dirty)
                self.edits[(fname, key)] = e
                grid.addWidget(e, row, col)
            row += 1

        root.addLayout(grid)

        # 时间微调
        time_layout = QHBoxLayout()
        time_layout.addStretch(1)

        lbl_cur = QLabel("本次时间调整（分钟）：")
        time_layout.addWidget(lbl_cur)

        self.spin_time_adjust = QDoubleSpinBox()
        self.spin_time_adjust.setDecimals(2)
        self.spin_time_adjust.setRange(-120.0, 120.0)
        self.spin_time_adjust.setSingleStep(1.0)
        self.spin_time_adjust.valueChanged.connect(self.mark_dirty)
        time_layout.addWidget(self.spin_time_adjust)

        time_layout.addSpacing(20)

        lbl_total_title = QLabel("当前总产线时间：")
        time_layout.addWidget(lbl_total_title)

        self.lbl_time_total = QLabel("0.00 分钟")
        time_layout.addWidget(self.lbl_time_total)

        time_layout.addSpacing(20)

        btn_clear_time = QPushButton("一键清除时间")
        btn_clear_time.clicked.connect(self.clear_time_adjust)
        time_layout.addWidget(btn_clear_time)

        time_layout.addStretch(1)
        root.addLayout(time_layout)

        # 底部按钮
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        btn_save = QPushButton("保存参数设置")
        btn_save.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.Bold))
        btn_save.clicked.connect(self.save_all)
        bottom.addWidget(btn_save)
        root.addLayout(bottom)

    # ---------- 功能槽 ----------
    def open_product_manager(self):
        dlg = ProductConfigManager(self)
        dlg.exec()

    def mark_dirty(self, *args):
        if self._loading:
            return
        self._dirty = True

    def _update_time_total_label(self):
        line = self.current_line
        info = self.params.get(line, {})
        try:
            total = float(info.get("time_total", info.get("time_adjust", 0.0)))
        except Exception:
            total = 0.0
        self.lbl_time_total.setText(f"{total:.2f} 分钟")

    def _set_furnace_widgets_enabled(self, fname: str, enabled: bool):
        for key in ("temperature", "air_flow", "salt"):
            w = self.edits[(fname, key)]
            w.setEnabled(enabled)

    def on_line_change(self, new_line: str):
        if self._loading:
            return

        old_line = self.current_line
        if new_line == old_line:
            return

        if self._dirty:
            msg = f"产线 {old_line} 的参数已修改，是否保存？"
            reply = QMessageBox.question(
                self, "提示", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.save_all()
                self._load_line(new_line)
            elif reply == QMessageBox.StandardButton.No:
                self._load_line(new_line)
            else:
                self._loading = True
                self.combo_line.setCurrentText(old_line)
                self._loading = False
                return
        else:
            self._load_line(new_line)

    def choose_log_path(self):
        path_str, _ = QFileDialog.getSaveFileName(
            self, "选择调整日志保存路径", str(self.log_path), "Excel 文件 (*.xlsx);;所有文件 (*.*)"
        )
        if path_str:
            self.log_path = Path(path_str)
            self.ed_log.setText(str(self.log_path))
            save_log_path(self.log_path)

    def on_enabled_changed(self, state: int):
        if self._loading:
            return
        sender = self.sender()
        fname_target = None
        for fname, chk in self.chk_enabled.items():
            if chk is sender:
                fname_target = fname
                break
        if fname_target is None:
            return

        enabled = (state == Qt.CheckState.Checked.value)
        self._set_furnace_widgets_enabled(fname_target, enabled)
        self.mark_dirty()

    def _load_line(self, line: str):
        self._loading = True
        self.current_line = line

        info = self.params.get(line, {})
        if "furnaces" not in info: info["furnaces"] = {}
        if "time_adjust" not in info: info["time_adjust"] = 0.0
        if "time_total" not in info:
            try:
                info["time_total"] = float(info.get("time_adjust", 0.0))
            except Exception:
                info["time_total"] = 0.0

        for fname in ("N2", "N3", "N4", "N5"):
            if fname not in info["furnaces"]:
                info["furnaces"][fname] = {"enabled": True, "temperature": None, "air_flow": None, "salt": None}
            else:
                if "enabled" not in info["furnaces"][fname]:
                    info["furnaces"][fname]["enabled"] = True

        self.params[line] = info
        furnaces = info.get("furnaces", {})
        for fname in ("N2", "N3", "N4", "N5"):
            fdata = furnaces.get(fname, {})
            enabled = bool(fdata.get("enabled", True))
            chk = self.chk_enabled[fname]
            chk.setChecked(enabled)
            for key in ("temperature", "air_flow", "salt"):
                w = self.edits[(fname, key)]
                val = fdata.get(key)
                w.setText("" if val is None else str(val))
            self._set_furnace_widgets_enabled(fname, enabled)

        self.spin_time_adjust.setValue(0.0)
        self._update_time_total_label()
        self._dirty = False
        self._loading = False

    def save_all(self):
        line = self.current_line
        info = self.params.get(line, {"furnaces": {}, "time_adjust": 0.0, "time_total": 0.0})
        if "furnaces" not in info: info["furnaces"] = {}

        # 旧数据备份
        old_info = json.loads(json.dumps(info, ensure_ascii=False))

        for fname in ("N2", "N3", "N4", "N5"):
            if fname not in info["furnaces"]:
                info["furnaces"][fname] = {"enabled": True, "temperature": None, "air_flow": None, "salt": None}
            fdata = info["furnaces"][fname]
            enabled = self.chk_enabled[fname].isChecked()
            fdata["enabled"] = bool(enabled)
            for key in ("temperature", "air_flow", "salt"):
                text = self.edits[(fname, key)].text().strip()
                if text == "":
                    fdata[key] = None
                else:
                    try:
                        fdata[key] = float(text)
                    except Exception:
                        QMessageBox.warning(self, "提示", f"{fname} 的 {key} 请输入数值")
                        return

        try:
            delta_time = float(self.spin_time_adjust.value())
        except Exception:
            delta_time = 0.0

        try:
            old_total = float(info.get("time_total", info.get("time_adjust", 0.0)))
        except Exception:
            old_total = 0.0

        new_total = old_total + delta_time
        info["time_total"] = new_total
        info["time_adjust"] = new_total
        self.params[line] = info

        try:
            save_params(self.params)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存参数文件失败：\n{e}")
            return

        try:
            self._log_changes(line, old_info, info)
        except Exception:
            traceback.print_exc()
            QMessageBox.warning(self, "提示", "参数保存成功，但日志记录出错。")

        self._loading = True
        self.spin_time_adjust.setValue(0.0)
        self._loading = False
        self._update_time_total_label()
        self._dirty = False
        QMessageBox.information(self, "完成", f"产线 {line} 参数已保存。")

    def clear_time_adjust(self):
        line = self.current_line
        info = self.params.get(line, {"furnaces": {}, "time_adjust": 0.0, "time_total": 0.0})
        try:
            old_total = float(info.get("time_total", info.get("time_adjust", 0.0)))
        except:
            old_total = 0.0

        if abs(old_total) < 1e-9:
            self.spin_time_adjust.setValue(0.0)
            self._update_time_total_label()
            return

        old_info = json.loads(json.dumps(info, ensure_ascii=False))
        info["time_total"] = 0.0
        info["time_adjust"] = 0.0
        self.params[line] = info

        try:
            save_params(self.params)
            self._log_changes(line, old_info, info)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"清除时间失败：\n{e}")
            return

        self._loading = True
        self.spin_time_adjust.setValue(0.0)
        self._loading = False
        self._update_time_total_label()
        self._dirty = False
        QMessageBox.information(self, "完成", f"产线 {line} 累计时间已清零。")

    def _log_changes(self, line: str, old: Dict[str, Any], new: Dict[str, Any]):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temp_rows, air_rows, salt_rows = [], [], []

        old_furn = (old.get("furnaces", {}) or {})
        new_furn = (new.get("furnaces", {}) or {})

        en_change_map = {}
        for fname in ("N2", "N3", "N4", "N5"):
            old_en = bool((old_furn.get(fname) or {}).get("enabled", True))
            new_en = bool((new_furn.get(fname) or {}).get("enabled", True))
            en_change_map[fname] = (old_en != new_en)
        global_en_changed = any(en_change_map.values())

        for fname in ("N2", "N3", "N4", "N5"):
            old_f = old_furn.get(fname, {}) or {}
            new_f = new_furn.get(fname, {}) or {}
            old_en = bool(old_f.get("enabled", True))
            new_en = bool(new_f.get("enabled", True))
            en_str = "启用" if new_en else "未启用"

            for key, kind in (("temperature", "temp"), ("air_flow", "air"), ("salt", "salt")):
                old_val = old_f.get(key)
                new_val = new_f.get(key)
                try:
                    old_flt = float(old_val) if old_val is not None else None
                except:
                    old_flt = None
                try:
                    new_flt = float(new_val) if new_val is not None else None
                except:
                    new_flt = None

                num_changed = (old_flt != new_flt)
                if not global_en_changed and not (en_change_map[fname] or num_changed):
                    continue

                if not new_en:
                    row = {"日期时间": now, "产线": line, "炉号": fname, "炉子是否启用": en_str, "调整前值": "-",
                           "调整量": "-", "调整后值": "-"}
                elif (not old_en) and new_en:
                    row = {"日期时间": now, "产线": line, "炉号": fname, "炉子是否启用": en_str, "调整前值": "-",
                           "调整量": "-", "调整后值": new_flt}
                else:
                    if num_changed and (new_flt is not None or old_flt is not None):
                        if old_flt is None and new_flt is not None:
                            delta = new_flt
                        elif new_flt is None and old_flt is not None:
                            delta = -old_flt
                        else:
                            delta = (new_flt - old_flt) if (new_flt is not None and old_flt is not None) else None
                        row = {"日期时间": now, "产线": line, "炉号": fname, "炉子是否启用": en_str,
                               "调整前值": old_flt, "调整量": delta, "调整后值": new_flt}
                    else:
                        row = {"日期时间": now, "产线": line, "炉号": fname, "炉子是否启用": en_str,
                               "调整前值": old_flt, "调整量": 0, "调整后值": new_flt}

                if kind == "temp":
                    temp_rows.append(row)
                elif kind == "air":
                    air_rows.append(row)
                elif kind == "salt":
                    salt_rows.append(row)

        if temp_rows: append_log_rows(self.log_path, pd.DataFrame(temp_rows), "温度调整", self)
        if air_rows: append_log_rows(self.log_path, pd.DataFrame(air_rows), "通气量调整", self)
        if salt_rows: append_log_rows(self.log_path, pd.DataFrame(salt_rows), "加盐量调整", self)

        # 时间调整日志
        try:
            old_t = float(old.get("time_total", old.get("time_adjust", 0.0)))
        except:
            old_t = 0.0
        try:
            new_t = float(new.get("time_total", new.get("time_adjust", 0.0)))
        except:
            new_t = 0.0

        if old_t != new_t:
            delta_t = new_t - old_t
            df_t = pd.DataFrame([{
                "日期时间": now, "产线": line, "调整前值": old_t, "调整量": delta_t,
                "调整后值": new_t, "总调整时间": new_t, "总产线水平时间": new_t
            }])
            append_log_rows(self.log_path, df_t, "时间调整", self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FurnaceManager()
    w.show()
    sys.exit(app.exec())