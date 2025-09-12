# ================================================================================
# 多曲線管理器 - multi_curve_manager.py
# 管理三條 FrequencyResponseData 實例
# ================================================================================

import numpy as np
from data_utils import FrequencyResponseData, Signal

class MultiCurveManager:
    """管理三條曲線：耳機響應、EQ調整、目標曲線"""
    
    def __init__(self):
        # 三條曲線
        self.headphone_curve = FrequencyResponseData()
        self.eq_curve = FrequencyResponseData()
        self.target_curve = FrequencyResponseData()
        
        # 狀態
        self.auto_calculate = True
        self.last_modified = None  # 最後修改的曲線
        
        # 信號
        self.curves_changed = Signal()
        self.calculation_done = Signal()
        
        # 連接信號
        self.headphone_curve.data_changed.connect(lambda: self.on_curve_changed('headphone'))
        self.eq_curve.data_changed.connect(lambda: self.on_curve_changed('eq'))
        self.target_curve.data_changed.connect(lambda: self.on_curve_changed('target'))
    
    def get_curve(self, curve_type):
        """獲取指定曲線"""
        if curve_type == 'headphone':
            return self.headphone_curve
        elif curve_type == 'eq':
            return self.eq_curve
        elif curve_type == 'target':
            return self.target_curve
        else:
            raise ValueError(f"未知曲線類型: {curve_type}")
    
    def on_curve_changed(self, curve_type):
        """曲線數據改變時的處理"""
        self.last_modified = curve_type
        
        if self.auto_calculate:
            self.calculate_missing_curve()
        
        self.curves_changed.emit()
    
    def calculate_missing_curve(self):
        """自動計算缺失的曲線"""
        # 檢查哪些曲線有數據
        has_headphone = len(self.headphone_curve.frequencies) > 0
        has_eq = len(self.eq_curve.frequencies) > 0
        has_target = len(self.target_curve.frequencies) > 0
        
        # 根據情況計算
        if has_headphone and has_eq and not has_target:
            # 計算目標曲線：耳機 + EQ = 目標
            self.calculate_target()
        elif has_target and has_headphone and not has_eq:
            # 計算EQ：目標 - 耳機 = EQ
            self.calculate_eq()
        elif has_target and has_eq and not has_headphone:
            # 計算耳機：目標 - EQ = 耳機
            self.calculate_headphone()
    
    def calculate_target(self):
        """計算目標曲線：耳機響應 + EQ調整 = 目標"""
        try:
            # 使用calculation_service計算
            from calculation_service import CurveCalculator
            result = CurveCalculator.add_curves(self.headphone_curve, self.eq_curve)
            
            if result:
                # 暫時斷開信號避免無限循環
                self.target_curve.data_changed.callbacks.clear()
                
                self.target_curve.frequencies = result['frequencies']
                self.target_curve.gains = result['gains']
                self.target_curve.metadata['calculated'] = True
                self.target_curve.metadata['formula'] = '耳機響應 + EQ調整'
                
                # 重新連接信號
                self.target_curve.data_changed.connect(lambda: self.on_curve_changed('target'))
                
                print("計算目標曲線完成")
                self.calculation_done.emit()
                
        except Exception as e:
            print(f"計算目標曲線錯誤: {e}")
    
    def calculate_eq(self):
        """計算EQ調整：目標 - 耳機響應 = EQ調整"""
        try:
            from calculation_service import CurveCalculator
            result = CurveCalculator.subtract_curves(self.target_curve, self.headphone_curve)
            
            if result:
                self.eq_curve.data_changed.callbacks.clear()
                
                self.eq_curve.frequencies = result['frequencies']
                self.eq_curve.gains = result['gains']
                self.eq_curve.metadata['calculated'] = True
                self.eq_curve.metadata['formula'] = '目標 - 耳機響應'
                
                self.eq_curve.data_changed.connect(lambda: self.on_curve_changed('eq'))
                
                print("計算EQ調整完成")
                self.calculation_done.emit()
                
        except Exception as e:
            print(f"計算EQ調整錯誤: {e}")
    
    def calculate_headphone(self):
        """計算耳機響應：目標 - EQ調整 = 耳機響應"""
        try:
            from calculation_service import CurveCalculator
            result = CurveCalculator.subtract_curves(self.target_curve, self.eq_curve)
            
            if result:
                self.headphone_curve.data_changed.callbacks.clear()
                
                self.headphone_curve.frequencies = result['frequencies']
                self.headphone_curve.gains = result['gains']
                self.headphone_curve.metadata['calculated'] = True
                self.headphone_curve.metadata['formula'] = '目標 - EQ調整'
                
                self.headphone_curve.data_changed.connect(lambda: self.on_curve_changed('headphone'))
                
                print("計算耳機響應完成")
                self.calculation_done.emit()
                
        except Exception as e:
            print(f"計算耳機響應錯誤: {e}")
    
    def load_preset_target(self, preset_name):
        """載入預設目標曲線"""
        try:
            if preset_name == 'harman':
                preset_data = FrequencyResponseData.create_harman_target()
            elif preset_name == 'flat':
                preset_data = FrequencyResponseData.create_flat_response()
            else:
                print(f"未知預設: {preset_name}")
                return False
            
            self.target_curve.copy_from(preset_data)
            print(f"已載入預設目標曲線: {preset_name}")
            return True
            
        except Exception as e:
            print(f"載入預設曲線錯誤: {e}")
            return False
    
    def clear_curve(self, curve_type):
        """清空指定曲線"""
        curve = self.get_curve(curve_type)
        curve.clear_data()
        print(f"已清空 {curve_type} 曲線")
    
    def clear_all(self):
        """清空所有曲線"""
        self.clear_curve('headphone')
        self.clear_curve('eq')
        self.clear_curve('target')
        self.last_modified = None
        print("已清空所有曲線")
    
    def get_status(self):
        """獲取狀態信息"""
        return {
            'headphone_points': len(self.headphone_curve.frequencies),
            'eq_points': len(self.eq_curve.frequencies),
            'target_points': len(self.target_curve.frequencies),
            'last_modified': self.last_modified,
            'auto_calculate': self.auto_calculate
        }