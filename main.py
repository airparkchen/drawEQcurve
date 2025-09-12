#!/usr/bin/env python3
# ================================================================================
# 頻率響應曲線編輯器 - 修正版本
# 解決 AttributeError: 'FrequencyPlotWidget' object has no attribute 'import_requested'
# ================================================================================

import sys
import os
import numpy as np
import re
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QDateTime
from PyQt5.QtGui import QFont, QMouseEvent
import pyqtgraph as pg
from scipy.interpolate import interp1d

# ================================================================================
# 1. 數據處理工具 (FrequencyResponseData)
# ================================================================================

class FrequencyResponseData(QObject):
    """
    頻率響應數據類別。
    儲存頻率和增益數據，並提供數據解析和插值功能。
    繼承 QObject 以使用 pyqtSignal。
    """
    
    data_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frequencies = np.array([])
        self.gains = np.array([])
        self.interpolated_frequencies = np.array([])
        self.interpolated_gains = np.array([])
        self.is_interpolated = False
        self.filename = None

    def parse_content(self, content):
        """解析文字內容，尋找頻率和增益數據"""
        try:
            # 兼容性：處理 freq, gain 格式
            if "FilterCurve:" not in content:
                lines = content.strip().split('\n')
                data = [list(map(float, line.split())) for line in lines if line.strip() and not line.strip().startswith('#')]
                data = np.array(data)
                
                if data.shape[1] < 2:
                    return False

                self.frequencies = data[:, 0]
                self.gains = data[:, 1]
                
            # 兼容性：處理 FilterCurve 格式
            else:
                freq_matches = re.findall(r'f(\d+)="([^"]+)"', content)
                gain_matches = re.findall(r'v(\d+)="([^"]+)"', content)

                freqs = {int(k): float(v) for k, v in freq_matches}
                gains = {int(k): float(v) for k, v in gain_matches}

                sorted_freqs = sorted(freqs.items())

                self.frequencies = np.array([v for k, v in sorted_freqs if k in gains])
                self.gains = np.array([gains[k] for k, v in sorted_freqs if k in gains])

            if len(self.frequencies) > 0:
                self.data_changed.emit()
                self.interpolate()  # 載入數據後自動插值
                return True
            else:
                return False
                
        except (ValueError, IndexError):
            return False

    def save_to_file(self, file_path):
        """將數據存檔到文字檔案"""
        try:
            # 簡單的頻率/增益格式
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Frequency (Hz), Gain (dB)\n")
                for freq, gain in zip(self.frequencies, self.gains):
                    f.write(f"{freq:.2f}\t{gain:+.2f}\n")
            return True
        except Exception:
            return False

    def update_gain_at_index(self, index, new_gain):
        """更新指定索引的增益值"""
        if 0 <= index < len(self.gains):
            self.gains[index] = new_gain
            self.data_changed.emit()
            self.interpolate()

    def interpolate(self, method='slinear'):
        """對數據進行插值"""
        if len(self.frequencies) < 2:
            self.is_interpolated = False
            self.interpolated_frequencies = np.array([])
            self.interpolated_gains = np.array([])
            self.data_changed.emit()
            return
        
        # 創建插值函數
        interp_func = interp1d(self.frequencies, self.gains, kind=method, bounds_error=False, fill_value="extrapolate")
        
        # 在指定範圍內生成新的頻率點
        new_frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)
        
        self.interpolated_frequencies = new_frequencies
        self.interpolated_gains = interp_func(new_frequencies)
        self.is_interpolated = True
        self.data_changed.emit()

    def clear(self):
        """清空所有數據"""
        self.frequencies = np.array([])
        self.gains = np.array([])
        self.interpolated_frequencies = np.array([])
        self.interpolated_gains = np.array([])
        self.is_interpolated = False
        self.filename = None
        self.data_changed.emit()

# ================================================================================
# 2. 曲線計算服務 (CurveCalculator)
# ================================================================================

class CurveCalculator:
    """處理曲線間的數學運算"""
    
    @staticmethod
    def align_frequencies(curve1, curve2):
        """對齊兩條曲線的頻率網格"""
        if len(curve1.frequencies) == 0 or len(curve2.frequencies) == 0:
            return None
        
        min_freq = min(curve1.frequencies[0], curve2.frequencies[0])
        max_freq = max(curve1.frequencies[-1], curve2.frequencies[-1])
        
        num_points = 200
        frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), num_points)
        
        return frequencies
    
    @staticmethod
    def interpolate_curve(curve, target_frequencies):
        """將曲線插值到目標頻率"""
        if len(curve.frequencies) == 0:
            return np.zeros_like(target_frequencies)
        
        try:
            gains = np.interp(target_frequencies, curve.frequencies, curve.gains)
            return gains
        except Exception:
            return np.zeros_like(target_frequencies)
    
    @staticmethod
    def add_curves(curve1, curve2):
        """兩條曲線相加：curve1 + curve2"""
        try:
            frequencies = CurveCalculator.align_frequencies(curve1, curve2)
            if frequencies is None:
                return None
            
            gains1 = CurveCalculator.interpolate_curve(curve1, frequencies)
            gains2 = CurveCalculator.interpolate_curve(curve2, frequencies)
            
            result_gains = gains1 + gains2
            
            return {
                'frequencies': frequencies,
                'gains': result_gains
            }
        except Exception:
            return None
    
    @staticmethod
    def subtract_curves(curve1, curve2):
        """兩條曲線相減：curve1 - curve2"""
        try:
            frequencies = CurveCalculator.align_frequencies(curve1, curve2)
            if frequencies is None:
                return None
            
            gains1 = CurveCalculator.interpolate_curve(curve1, frequencies)
            gains2 = CurveCalculator.interpolate_curve(curve2, frequencies)
            
            result_gains = gains1 - gains2
            
            return {
                'frequencies': frequencies,
                'gains': result_gains
            }
        except Exception:
            return None

# ================================================================================
# 3. 多曲線管理器 (MultiCurveManager)
# ================================================================================

class MultiCurveManager(QObject):
    """管理三條頻率響應曲線並處理它們之間的關係"""
    
    curves_changed = pyqtSignal()
    calculation_done = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.headphone_curve = FrequencyResponseData(self)
        self.eq_curve = FrequencyResponseData(self)
        self.target_curve = FrequencyResponseData(self)
        
        self.last_modified = None
        self.auto_calculate = True
        
        self.headphone_curve.data_changed.connect(lambda: self.handle_curve_change('headphone'))
        self.eq_curve.data_changed.connect(lambda: self.handle_curve_change('eq'))
        self.target_curve.data_changed.connect(lambda: self.handle_curve_change('target'))
        
    def get_curve(self, curve_type):
        """根據類型返回對應的曲線物件"""
        if curve_type == 'headphone':
            return self.headphone_curve
        elif curve_type == 'eq':
            return self.eq_curve
        elif curve_type == 'target':
            return self.target_curve
        else:
            raise ValueError("無效的曲線類型")
            
    def handle_curve_change(self, curve_type):
        """當任一曲線數據改變時被呼叫"""
        self.last_modified = f"{curve_type} at {QDateTime.currentDateTime().toString('hh:mm:ss')}"
        if self.auto_calculate:
            self.calculate_eq() # 簡化為只計算 EQ
        self.curves_changed.emit()

    def calculate_eq(self):
        """計算 EQ 調整曲線 (目標 - 耳機)"""
        if len(self.headphone_curve.frequencies) == 0 or len(self.target_curve.frequencies) == 0:
            return
        
        result = CurveCalculator.subtract_curves(self.target_curve, self.headphone_curve)
        if result:
            self.eq_curve.frequencies = result['frequencies']
            self.eq_curve.gains = result['gains']
            self.eq_curve.is_interpolated = False
            self.calculation_done.emit()
            
    def calculate_headphone(self):
        """計算耳機響應曲線 (EQ + 目標)"""
        if len(self.eq_curve.frequencies) == 0 or len(self.target_curve.frequencies) == 0:
            return
            
        result = CurveCalculator.subtract_curves(self.target_curve, self.eq_curve)
        if result:
            self.headphone_curve.frequencies = result['frequencies']
            self.headphone_curve.gains = result['gains']
            self.headphone_curve.is_interpolated = False
            self.calculation_done.emit()
    
    def calculate_target(self):
        """計算目標曲線 (耳機 - EQ)"""
        if len(self.headphone_curve.frequencies) == 0 or len(self.eq_curve.frequencies) == 0:
            return
        
        result = CurveCalculator.subtract_curves(self.headphone_curve, self.eq_curve)
        if result:
            self.target_curve.frequencies = result['frequencies']
            self.target_curve.gains = result['gains']
            self.target_curve.is_interpolated = False
            self.calculation_done.emit()

    def load_preset_target(self, preset_type):
        """載入預設目標曲線"""
        if preset_type == 'harman':
            freqs = [20, 100, 500, 1000, 2000, 5000, 10000, 20000]
            gains = [3, 1, 0, -2, -4, -6, -8, -10]
        elif preset_type == 'flat':
            freqs = [20, 20000]
            gains = [0, 0]
        else:
            return
            
        self.target_curve.frequencies = np.array(freqs)
        self.target_curve.gains = np.array(gains)
        self.target_curve.filename = f"{preset_type}_preset.txt"
        self.target_curve.data_changed.emit()

    def clear_all(self):
        """清空所有曲線數據"""
        self.headphone_curve.clear()
        self.eq_curve.clear()
        self.target_curve.clear()
        self.last_modified = None
        self.curves_changed.emit()

    def get_status(self):
        """取得目前系統狀態"""
        return {
            'headphone_points': len(self.headphone_curve.frequencies),
            'eq_points': len(self.eq_curve.frequencies),
            'target_points': len(self.target_curve.frequencies),
            'last_modified': self.last_modified,
            'auto_calculate': self.auto_calculate
        }

# ================================================================================
# 4. 繪圖組件 (FrequencyPlotWidget)
# ================================================================================

class FrequencyPlotWidget(QWidget):
    """頻率響應繪圖組件（修正自原版）"""
    
    point_selected = pyqtSignal(int)
    
    def __init__(self, data, title="頻率響應", color='#2196F3'):
        super().__init__()
        self.data = data
        self.title = title
        self.color = color
        self.selected_index = None
        self.dragging = False
        self.show_original = True
        self.show_interpolated = True
        
        self.setup_ui()
        self.data.data_changed.connect(self.update_plot)
        
    def setup_ui(self):
        """設定界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 11, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.color}; padding: 5px;")
        layout.addWidget(title_label)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setMinimumHeight(200)
        layout.addWidget(self.plot_widget)
        
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setMenuEnabled(False)
        self.plot_item.showButtons = False
        viewbox = self.plot_item.getViewBox()
        viewbox.setMouseEnabled(x=False, y=False)
        viewbox.setMenuEnabled(False)
        viewbox.enableAutoRange(enable=False)
        
        self.plot_item.setXRange(0, 10000, padding=0)
        self.plot_item.setYRange(-20, 20, padding=0)
        self.plot_item.setLabel('left', '增益 (dB)')
        self.plot_item.setLabel('bottom', '頻率 (Hz)')
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        
        self.setup_nonlinear_x_axis()
        
        y_ticks = list(range(-20, 25, 5))
        y_labels = [f"{y:+d}" if y != 0 else "0" for y in y_ticks]
        y_axis = self.plot_item.getAxis('left')
        y_axis.setTicks([[(tick, label) for tick, label in zip(y_ticks, y_labels)]])
        
        self.plot_item.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        
        self.original_line = None
        self.original_points = None
        self.interpolated_line = None
        self.selected_point = None
        
        self.plot_widget.mousePressEvent = self.mouse_press_event
        self.plot_widget.mouseMoveEvent = self.mouse_move_event
        self.plot_widget.mouseReleaseEvent = self.mouse_release_event
        
    def setup_nonlinear_x_axis(self):
        """設定非線性X軸"""
        def freq_to_display(freq):
            if freq <= 100: return freq * 25.0
            elif freq <= 1000: return 2500 + (freq - 100) * (3000 / 900)
            elif freq <= 10000: return 5500 + (freq - 1000) * (3000 / 9000)
            else: return 8500 + (freq - 10000) * (1500 / 10000)
        
        def display_to_freq(display_pos):
            if display_pos <= 2500: return display_pos / 25.0
            elif display_pos <= 5500: return 100 + (display_pos - 2500) * (900 / 3000)
            elif display_pos <= 8500: return 1000 + (display_pos - 5500) * (9000 / 3000)
            else: return 10000 + (display_pos - 8500) * (10000 / 1500)
        
        self.freq_to_display = freq_to_display
        self.display_to_freq = display_to_freq
        
        major_freqs = [20, 50, 100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000, 20000]
        major_labels = ['20', '50', '100', '200', '300', '500', '700', '1k', '2k', '3k', '5k', '7k', '10k', '20k']
        
        major_display_pos = [freq_to_display(freq) for freq in major_freqs]
        x_axis = self.plot_item.getAxis('bottom')
        x_axis.setTicks([[(pos, label) for pos, label in zip(major_display_pos, major_labels)]])
        self.plot_item.setXRange(self.freq_to_display(20), self.freq_to_display(20000), padding=0)
        
    def transform_frequency_data(self, frequencies):
        """將實際頻率轉換為顯示位置"""
        return np.array([self.freq_to_display(freq) for freq in frequencies])
    
    def update_plot(self):
        """更新圖表"""
        for item in [self.original_line, self.original_points, self.interpolated_line, self.selected_point]:
            if item: self.plot_item.removeItem(item)
        
        if len(self.data.frequencies) == 0:
            return
        
        display_frequencies = self.transform_frequency_data(self.data.frequencies)
        
        if self.show_original:
            self.original_line = self.plot_item.plot(
                display_frequencies, self.data.gains, pen=pg.mkPen(self.color, width=2)
            )
            self.original_points = pg.ScatterPlotItem(
                pos=list(zip(display_frequencies, self.data.gains)),
                size=15, brush=pg.mkBrush(self.color), pen=pg.mkPen('white', width=3), symbol='o'
            )
            self.plot_item.addItem(self.original_points)
            
        if self.show_interpolated and self.data.is_interpolated:
            display_interp_frequencies = self.transform_frequency_data(self.data.interpolated_frequencies)
            self.interpolated_line = self.plot_item.plot(
                display_interp_frequencies, self.data.interpolated_gains,
                pen=pg.mkPen('#4CAF50', width=2)
            )
        
        if self.selected_index is not None and self.show_original:
            freq = self.data.frequencies[self.selected_index]
            gain = self.data.gains[self.selected_index]
            display_freq = self.freq_to_display(freq)
            
            self.selected_point = pg.ScatterPlotItem(
                pos=[(display_freq, gain)],
                size=20, brush=pg.mkBrush('#FF9800'), pen=pg.mkPen('white', width=4), symbol='o'
            )
            self.plot_item.addItem(self.selected_point)
    
    def mouse_press_event(self, event: QMouseEvent):
        """滑鼠按下"""
        if event.button() != Qt.LeftButton or len(self.data.frequencies) == 0:
            return
            
        pos = event.pos()
        view_pos = self.plot_item.vb.mapDeviceToView(pos)
        actual_freq = self.display_to_freq(view_pos.x())
        self.selected_index = self.find_nearest_point(actual_freq, view_pos.y())
        
        if self.selected_index is not None:
            self.dragging = True
            self.update_plot()
            self.point_selected.emit(self.selected_index)
    
    def mouse_move_event(self, event: QMouseEvent):
        """滑鼠移動"""
        if not self.dragging or self.selected_index is None:
            return
            
        pos = event.pos()
        view_pos = self.plot_item.vb.mapDeviceToView(pos)
        new_gain = max(-20, min(20, view_pos.y()))
        self.data.update_gain_at_index(self.selected_index, new_gain)
    
    def mouse_release_event(self, event: QMouseEvent):
        """滑鼠釋放"""
        self.dragging = False
    
    def find_nearest_point(self, actual_freq, y):
        """找最近的點"""
        if len(self.data.frequencies) == 0:
            return None
        
        freq_distances = (self.data.frequencies - actual_freq) / 25000
        gain_distances = (self.data.gains - y) / 40
        distances = np.sqrt(freq_distances**2 + gain_distances**2)
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] < 0.03:
            return nearest_idx
        return None
    
    def update_display(self, show_original, show_interpolated):
        """更新顯示選項"""
        self.show_original = show_original
        self.show_interpolated = show_interpolated
        self.update_plot()

# ================================================================================
# 5. 控制面板 (ControlPanel)
# ================================================================================

class ControlPanel(QWidget):
    """控制面板"""
    
    display_changed = pyqtSignal(bool, bool)
    
    def __init__(self, curve_manager):
        super().__init__()
        self.manager = curve_manager
        self.setup_ui()
        self.manager.curves_changed.connect(self.update_info)
        self.manager.calculation_done.connect(self.update_info)
        
    def setup_ui(self):
        """設定控制面板"""
        layout = QVBoxLayout(self)
        
        calc_group = QGroupBox("計算控制")
        calc_layout = QVBoxLayout(calc_group)
        
        self.auto_calc_cb = QCheckBox("自動計算")
        self.auto_calc_cb.setChecked(True)
        self.auto_calc_cb.toggled.connect(self.on_auto_calc_changed)
        calc_layout.addWidget(self.auto_calc_cb)
        
        calc_target_btn = QPushButton("計算目標曲線 (耳機-EQ)")
        calc_target_btn.clicked.connect(lambda: self.manager.calculate_target())
        calc_layout.addWidget(calc_target_btn)
        
        calc_eq_btn = QPushButton("計算EQ調整 (目標-耳機)")
        calc_eq_btn.clicked.connect(lambda: self.manager.calculate_eq())
        calc_layout.addWidget(calc_eq_btn)
        
        calc_headphone_btn = QPushButton("計算耳機響應 (目標+EQ)")
        calc_headphone_btn.clicked.connect(lambda: self.manager.calculate_headphone())
        calc_layout.addWidget(calc_headphone_btn)
        
        layout.addWidget(calc_group)
        
        preset_group = QGroupBox("預設曲線")
        preset_layout = QVBoxLayout(preset_group)
        
        harman_btn = QPushButton("載入 Harman 目標")
        harman_btn.clicked.connect(lambda: self.manager.load_preset_target('harman'))
        preset_layout.addWidget(harman_btn)
        
        flat_btn = QPushButton("載入平坦響應")
        flat_btn.clicked.connect(lambda: self.manager.load_preset_target('flat'))
        preset_layout.addWidget(flat_btn)
        
        layout.addWidget(preset_group)
        
        display_group = QGroupBox("顯示選項")
        display_layout = QVBoxLayout(display_group)
        
        self.show_original_cb = QCheckBox("顯示原始數據點")
        self.show_original_cb.setChecked(True)
        self.show_original_cb.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_original_cb)
        
        self.show_interpolated_cb = QCheckBox("顯示插值曲線")
        self.show_interpolated_cb.setChecked(True)
        self.show_interpolated_cb.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_interpolated_cb)
        
        layout.addWidget(display_group)
        
        info_group = QGroupBox("系統狀態")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setStyleSheet("QTextEdit { border: 1px solid #E0E0E0; border-radius: 4px; padding: 5px; background-color: #FAFAFA; font-family: monospace; font-size: 10px; }")
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        clear_btn = QPushButton("清除所有曲線")
        clear_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; border: none; border-radius: 4px; padding: 8px; font-weight: bold; } QPushButton:hover { background-color: #d32f2f; }")
        clear_btn.clicked.connect(self.clear_all_safe)
        layout.addWidget(clear_btn)
        
        layout.addStretch()
        
    def clear_all_safe(self):
        """安全清除所有曲線"""
        reply = QMessageBox.question(self, '確認', '確定要清除所有曲線嗎？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.manager.clear_all()
        
    def on_display_changed(self):
        """顯示選項改變"""
        show_original = self.show_original_cb.isChecked()
        show_interpolated = self.show_interpolated_cb.isChecked()
        self.display_changed.emit(show_original, show_interpolated)
        
    def on_auto_calc_changed(self, checked):
        """自動計算開關變化"""
        self.manager.auto_calculate = checked
        
    def update_info(self):
        """更新檔案資訊"""
        status = self.manager.get_status()
        info = [
            f"耳機響應: {status['headphone_points']} 個點",
            f"EQ調整: {status['eq_points']} 個點",
            f"目標曲線: {status['target_points']} 個點",
            f"最後修改: {status['last_modified'] or '無'}",
            f"自動計算: {'開啟' if status['auto_calculate'] else '關閉'}"
        ]
        self.info_text.setText("\n".join(info))

# ================================================================================
# 6. 主視窗 (FrequencyResponseEditor)
# ================================================================================

class FrequencyResponseEditor(QMainWindow):
    """主視窗"""
    
    def __init__(self):
        super().__init__()
        
        self.curve_manager = MultiCurveManager()
        
        self.setWindowTitle("三曲線EQ調校系統 - 頻率響應曲線編輯器")
        self.resize(1400, 900)
        self.setStyleSheet("QMainWindow { background-color: #FAFAFA; }")
        
        self.setup_ui()
        self.setup_menu()
        self.connect_signals()
        self.load_sample_data()
        
    def setup_ui(self):
        """設定界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        self.headphone_plot = FrequencyPlotWidget(
            self.curve_manager.headphone_curve, "🎧 耳機響應曲線", '#E91E63'
        )
        left_layout.addWidget(self.headphone_plot)
        
        self.eq_plot = FrequencyPlotWidget(
            self.curve_manager.eq_curve, "🎛️ EQ調整曲線", '#2196F3'
        )
        left_layout.addWidget(self.eq_plot)
        
        self.target_plot = FrequencyPlotWidget(
            self.curve_manager.target_curve, "🎯 目標曲線", '#4CAF50'
        )
        left_layout.addWidget(self.target_plot)
        
        layout.addWidget(left_widget, stretch=3)
        
        self.control_panel = ControlPanel(self.curve_manager)
        layout.addWidget(self.control_panel, stretch=1)
        
    def setup_menu(self):
        """設定選單"""
        menubar = self.menuBar()
        file_menu = menubar.addMenu('檔案')
        
        import_menu = file_menu.addMenu('匯入')
        import_headphone_action = QAction('匯入耳機響應...', self)
        import_headphone_action.triggered.connect(lambda: self.import_file('headphone'))
        import_menu.addAction(import_headphone_action)
        
        import_eq_action = QAction('匯入EQ調整...', self)
        import_eq_action.triggered.connect(lambda: self.import_file('eq'))
        import_menu.addAction(import_eq_action)
        
        import_target_action = QAction('匯入目標曲線...', self)
        import_target_action.triggered.connect(lambda: self.import_file('target'))
        import_menu.addAction(import_target_action)

        export_menu = file_menu.addMenu('匯出')
        export_headphone_action = QAction('匯出耳機響應...', self)
        export_headphone_action.triggered.connect(lambda: self.export_file('headphone'))
        export_menu.addAction(export_headphone_action)

        export_eq_action = QAction('匯出EQ調整...', self)
        export_eq_action.triggered.connect(lambda: self.export_file('eq'))
        export_menu.addAction(export_eq_action)

        export_target_action = QAction('匯出目標曲線...', self)
        export_target_action.triggered.connect(lambda: self.export_file('target'))
        export_menu.addAction(export_target_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('結束', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        self.statusBar().showMessage("就緒 - 三曲線EQ調校系統")
        
    def connect_signals(self):
        """連接信號"""
        self.headphone_plot.point_selected.connect(lambda idx: self.on_point_selected(idx, 'headphone'))
        self.eq_plot.point_selected.connect(lambda idx: self.on_point_selected(idx, 'eq'))
        self.target_plot.point_selected.connect(lambda idx: self.on_point_selected(idx, 'target'))
        
        self.control_panel.display_changed.connect(self.update_all_displays)
        self.curve_manager.curves_changed.connect(self.update_all_plots)
        self.curve_manager.calculation_done.connect(self.update_all_plots)
        
    def update_all_displays(self, show_original, show_interpolated):
        """更新所有面板顯示"""
        self.headphone_plot.update_display(show_original, show_interpolated)
        self.eq_plot.update_display(show_original, show_interpolated)
        self.target_plot.update_display(show_original, show_interpolated)
        
    def update_all_plots(self):
        """更新所有圖表"""
        self.headphone_plot.update_plot()
        self.eq_plot.update_plot()
        self.target_plot.update_plot()
        
    def on_point_selected(self, index, curve_type):
        """點被選中"""
        curve = self.curve_manager.get_curve(curve_type)
        if len(curve.frequencies) > index:
            freq = curve.frequencies[index]
            gain = curve.gains[index]
            self.statusBar().showMessage(f"選中 {curve_type}: {freq:.0f}Hz, {gain:+.2f}dB")
        
    def import_file(self, curve_type):
        """匯入檔案功能"""
        curve_names = {'headphone': '耳機響應曲線', 'eq': 'EQ調整曲線', 'target': '目標曲線'}
        file_path, _ = QFileDialog.getOpenFileName(self, f"匯入{curve_names[curve_type]}", "", "文字檔案 (*.txt);;所有檔案 (*.*)")
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                curve = self.curve_manager.get_curve(curve_type)
                if curve.parse_content(content):
                    curve.filename = os.path.basename(file_path)
                    self.statusBar().showMessage(f"已匯入{curve_names[curve_type]}: {curve.filename}", 3000)
                else:
                    QMessageBox.critical(self, "錯誤", "無法解析檔案格式")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法匯入檔案: {str(e)}")
    
    def export_file(self, curve_type):
        """匯出檔案功能"""
        curve_names = {'headphone': '耳機響應曲線', 'eq': 'EQ調整曲線', 'target': '目標曲線'}
        curve = self.curve_manager.get_curve(curve_type)
        if len(curve.frequencies) == 0:
            QMessageBox.warning(self, "警告", f"{curve_names[curve_type]}無數據可匯出")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, f"匯出{curve_names[curve_type]}", curve.filename or f"{curve_type}_curve.txt", "文字檔案 (*.txt);;所有檔案 (*.*)")
        
        if file_path:
            if curve.save_to_file(file_path):
                curve.filename = os.path.basename(file_path)
                self.statusBar().showMessage(f"已匯出{curve_names[curve_type]}: {curve.filename}", 3000)
            else:
                QMessageBox.critical(self, "錯誤", "無法匯出檔案")
            
    def load_sample_data(self):
        """載入範例數據"""
        sample_content = '''FilterCurve:f0="20" f1="30" f2="40" f3="50" f4="60" f5="70" f6="80" f7="90" f8="100" f9="135" f10="170" f11="200" f12="250" f13="300" f14="400" f15="500" f16="600" f17="700" f18="800" f19="900" f20="1000" f21="2000" f22="3000" f23="4000" f24="5000" f25="6000" f26="7000" f27="8000" f28="9000" f29="10000" f30="12500" f31="15000" f32="17500" f33="20000" v0="-2" v1="-4.333334" v2="-4.1435184" v3="-5.8518524" v4="-5.8518524" v5="-6.041666" v6="-5.2824078" v7="-5.6620369" v8="-5.8518524" v9="-3.9537029" v10="-3.3842583" v11="-1.8657417" v12="-1.2962971" v13="0.22222137" v14="-0.15740776" v15="0.032407761" v16="0.4120369" v17="0.22222137" v18="0.22222137" v19="-0.15740776" v20="0.4120369" v21="5.3472233" v22="5.3472233" v23="6.1064816" v24="6.8657417" v25="6.8657417" v26="5.9166679" v27="0.032407761" v28="-1.2962971" v29="-3.1944447" v30="-2.245369" v31="-0.5370369" v32="0.60185242" v33="2.5" FilterLength="8191"'''
        
        if self.curve_manager.eq_curve.parse_content(sample_content):
            self.curve_manager.eq_curve.filename = "範例EQ調整.txt"
            self.statusBar().showMessage("已載入範例EQ調整數據", 3000)

# ================================================================================
# 主程式入口
# ================================================================================

def main():
    """主程式"""
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    
    window = FrequencyResponseEditor()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()