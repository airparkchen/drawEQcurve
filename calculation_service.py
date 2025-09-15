# ================================================================================
# 曲線計算服務 - calculation_service.py  
# 處理曲線間的數學運算
# 修正版：支援完整的三曲線計算關係
# ================================================================================

import numpy as np
from scipy.interpolate import interp1d

class CurveCalculator:
    """曲線計算工具"""
    
    @staticmethod
    def align_frequencies(curve1, curve2):
        """對齊兩條曲線的頻率網格"""
        if len(curve1.frequencies) == 0 or len(curve2.frequencies) == 0:
            return None
        
        # 找到共同的頻率範圍
        min_freq = max(curve1.frequencies[0], curve2.frequencies[0])
        max_freq = min(curve1.frequencies[-1], curve2.frequencies[-1])
        
        if min_freq >= max_freq:
            print("警告：兩條曲線沒有重疊的頻率範圍")
            # 使用更大的範圍
            min_freq = min(curve1.frequencies[0], curve2.frequencies[0])
            max_freq = max(curve1.frequencies[-1], curve2.frequencies[-1])
        
        # 生成統一的頻率網格（對數分布）
        num_points = 200
        frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), num_points)
        
        return frequencies
    
    @staticmethod
    def interpolate_curve(curve, target_frequencies):
        """將曲線插值到目標頻率"""
        if len(curve.frequencies) == 0:
            return np.zeros_like(target_frequencies)
        
        try:
            # 使用線性插值（簡單可靠）
            gains = np.interp(target_frequencies, curve.frequencies, curve.gains)
            return gains
        except Exception as e:
            print(f"插值錯誤: {e}")
            return np.zeros_like(target_frequencies)
    
    @staticmethod
    def add_curves(curve1, curve2):
        """兩條曲線相加：curve1 + curve2
        主要用於：耳機響應 + EQ調整 = 目標曲線"""
        try:
            # 對齊頻率
            frequencies = CurveCalculator.align_frequencies(curve1, curve2)
            if frequencies is None:
                return None
            
            # 插值到統一頻率
            gains1 = CurveCalculator.interpolate_curve(curve1, frequencies)
            gains2 = CurveCalculator.interpolate_curve(curve2, frequencies)
            
            # 相加
            result_gains = gains1 + gains2
            
            return {
                'frequencies': frequencies,
                'gains': result_gains
            }
            
        except Exception as e:
            print(f"曲線相加錯誤: {e}")
            return None
    
    @staticmethod
    def subtract_curves(curve1, curve2):
        """兩條曲線相減：curve1 - curve2
        主要用於：
        - 目標曲線 - 耳機響應 = EQ調整
        - 目標曲線 - EQ調整 = 耳機響應"""
        try:
            # 對齊頻率
            frequencies = CurveCalculator.align_frequencies(curve1, curve2)
            if frequencies is None:
                return None
            
            # 插值到統一頻率
            gains1 = CurveCalculator.interpolate_curve(curve1, frequencies)
            gains2 = CurveCalculator.interpolate_curve(curve2, frequencies)
            
            # 相減
            result_gains = gains1 - gains2
            
            return {
                'frequencies': frequencies,
                'gains': result_gains
            }
            
        except Exception as e:
            print(f"曲線相減錯誤: {e}")
            return None
    
    # ========================================================================
    # 新增：三曲線專用計算方法
    # ========================================================================
    
    @staticmethod
    def calculate_target_from_headphone_eq(headphone, eq):
        """計算目標曲線：耳機響應 + EQ調整 = 目標曲線"""
        return CurveCalculator.add_curves(headphone, eq)
    
    @staticmethod
    def calculate_eq_from_target_headphone(target, headphone):
        """計算EQ調整：目標曲線 - 耳機響應 = EQ調整"""
        return CurveCalculator.subtract_curves(target, headphone)
    
    @staticmethod
    def calculate_headphone_from_target_eq(target, eq):
        """計算耳機響應：目標曲線 - EQ調整 = 耳機響應"""
        return CurveCalculator.subtract_curves(target, eq)
    
    # ========================================================================
    # 原有方法保持不變
    # ========================================================================
    
    @staticmethod
    def validate_curve_relationship(headphone, eq, target, tolerance=1.0):
        """驗證三條曲線的數學關係：耳機響應 + EQ調整 = 目標曲線"""
        try:
            # 計算理論目標曲線
            calculated_target = CurveCalculator.add_curves(headphone, eq)
            if not calculated_target:
                return {'valid': False, 'error': '無法計算理論目標曲線'}
            
            # 插值實際目標曲線到相同頻率
            actual_gains = CurveCalculator.interpolate_curve(target, calculated_target['frequencies'])
            
            # 計算差異
            differences = np.abs(calculated_target['gains'] - actual_gains)
            max_diff = np.max(differences)
            mean_diff = np.mean(differences)
            
            is_valid = max_diff <= tolerance
            
            return {
                'valid': is_valid,
                'max_difference': max_diff,
                'mean_difference': mean_diff,
                'tolerance': tolerance,
                'message': f"最大偏差: {max_diff:.2f} dB" + 
                          (f" (超出容差 {tolerance} dB)" if not is_valid else " (符合要求)")
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'驗證錯誤: {e}'}
    
    @staticmethod
    def smooth_curve(frequencies, gains, window_size=5):
        """平滑曲線"""
        if len(gains) < window_size:
            return gains
        
        try:
            # 簡單移動平均
            smoothed = np.convolve(gains, np.ones(window_size)/window_size, mode='same')
            return smoothed
        except Exception as e:
            print(f"平滑錯誤: {e}")
            return gains
    
    @staticmethod
    def resample_curve(curve, new_frequencies):
        """重新採樣曲線到新的頻率點"""
        if len(curve.frequencies) == 0:
            return {
                'frequencies': new_frequencies,
                'gains': np.zeros_like(new_frequencies)
            }
        
        try:
            new_gains = CurveCalculator.interpolate_curve(curve, new_frequencies)
            return {
                'frequencies': new_frequencies,
                'gains': new_gains
            }
        except Exception as e:
            print(f"重新採樣錯誤: {e}")
            return None