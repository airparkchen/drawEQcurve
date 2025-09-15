# ================================================================================
# 數據處理工具 - data_utils.py
# 處理頻率響應數據的解析、插值、儲存
# ================================================================================

import numpy as np
import re

class Signal:
    """簡化的信號系統"""
    def __init__(self):
        self.callbacks = []
    
    def connect(self, callback):
        self.callbacks.append(callback)
    
    def emit(self, *args, **kwargs):
        for callback in self.callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"信號回調錯誤: {e}")

class FrequencyResponseData:
    """頻率響應數據模型"""
    
    def __init__(self):
        self.frequencies = np.array([])
        self.gains = np.array([])
        self.metadata = {}
        self.interpolated_frequencies = np.array([])
        self.interpolated_gains = np.array([])
        self.is_interpolated = False
        self.filename = ""
        self.is_modified = False
        
        # 信號
        self.data_changed = Signal()
        self.interpolation_changed = Signal()
    
    def parse_content(self, content: str) -> bool:
        """解析檔案內容"""
        try:
            # 移除 FilterCurve: 前綴
            if content.startswith('FilterCurve:'):
                content = content[12:]
            
            # 解析所有 key="value" 配對
            pattern = r'(\w+)="([^"]*)"'
            matches = re.findall(pattern, content)
            
            freq_dict = {}
            gain_dict = {}
            
            for key, value in matches:
                if key.startswith('f') and key[1:].isdigit():
                    # 頻率數據: f0, f1, f2, ...
                    index = int(key[1:])
                    freq_dict[index] = float(value)
                elif key.startswith('v') and key[1:].isdigit():
                    # 增益數據: v0, v1, v2, ...
                    index = int(key[1:])
                    gain_dict[index] = float(value)
                else:
                    # 元數據
                    self.metadata[key] = value
            
            # 按索引排序並轉換為陣列
            freq_indices = sorted(freq_dict.keys())
            gain_indices = sorted(gain_dict.keys())
            
            # 驗證數據完整性
            if freq_indices != gain_indices:
                raise ValueError("頻率和增益數據索引不匹配")
            
            self.frequencies = np.array([freq_dict[i] for i in freq_indices])
            self.gains = np.array([gain_dict[i] for i in gain_indices])
            
            # 驗證頻率遞增
            if not np.all(np.diff(self.frequencies) > 0):
                raise ValueError("頻率數據必須遞增")
            
            print(f"成功解析數據: {len(self.frequencies)} 個點")
            print(f"頻率範圍: {self.frequencies[0]} - {self.frequencies[-1]} Hz")
            
            self.data_changed.emit()
            return True
            
        except Exception as e:
            print(f"解析錯誤: {e}")
            return False
    
    def interpolate(self, method: str = 'cubic', num_points: int = 500) -> bool:
        """執行插值 - 生成經過所有原始點的平滑曲線"""
        if len(self.frequencies) < 2:
            print("需要至少2個數據點才能插值")
            return False
            
        try:
            # 獲取原始數據範圍
            freq_min = float(self.frequencies[0])
            freq_max = float(self.frequencies[-1])
            
            print(f"開始插值: {method}, {num_points} 點")
            print(f"原始範圍: {freq_min} - {freq_max} Hz")
            
            # 創建密集的插值頻率點（對數分布更符合音頻特性）
            self.interpolated_frequencies = np.logspace(
                np.log10(freq_min), np.log10(freq_max), num_points
            )
            
            # 根據方法選擇插值演算法
            if method == 'linear':
                # 線性插值（最簡單）
                self.interpolated_gains = np.interp(
                    self.interpolated_frequencies,
                    self.frequencies,
                    self.gains
                )
                print("使用線性插值")
                
            elif method == 'cubic':
                # 三次樣條插值（經過所有點的平滑曲線）
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(self.frequencies, self.gains, bc_type='natural')
                self.interpolated_gains = cs(self.interpolated_frequencies)
                print("使用三次樣條插值（平滑曲線）")
                
            elif method == 'quadratic':
                # 二次樣條插值
                from scipy.interpolate import interp1d
                if len(self.frequencies) >= 3:
                    f = interp1d(self.frequencies, self.gains, kind='quadratic', 
                               bounds_error=False, fill_value='extrapolate')
                    self.interpolated_gains = f(self.interpolated_frequencies)
                    print("使用二次樣條插值")
                else:
                    # 點數不足，降級為線性
                    self.interpolated_gains = np.interp(
                        self.interpolated_frequencies,
                        self.frequencies,
                        self.gains
                    )
                    print("點數不足，使用線性插值")
            else:
                # 默認線性插值
                self.interpolated_gains = np.interp(
                    self.interpolated_frequencies,
                    self.frequencies,
                    self.gains
                )
                print("使用默認線性插值")
            
            self.is_interpolated = True
            self.interpolation_changed.emit()
            
            print(f"插值成功完成！")
            print(f"原始點數: {len(self.frequencies)}")
            print(f"插值點數: {len(self.interpolated_frequencies)}")
            return True
            
        except Exception as e:
            print(f"插值錯誤: {e}")
            # 嘗試最簡單的線性插值作為後備
            try:
                print("嘗試後備線性插值...")
                self.interpolated_frequencies = np.logspace(
                    np.log10(self.frequencies[0]),
                    np.log10(self.frequencies[-1]),
                    num_points
                )
                self.interpolated_gains = np.interp(
                    self.interpolated_frequencies,
                    self.frequencies,
                    self.gains
                )
                self.is_interpolated = True
                self.interpolation_changed.emit()
                print("後備插值成功")
                return True
            except Exception as backup_e:
                print(f"後備插值也失敗: {backup_e}")
                self.is_interpolated = False
                return False
    
    def update_gain_at_index(self, index: int, new_gain: float):
        """更新指定索引的增益值"""
        if 0 <= index < len(self.gains):
            self.gains[index] = new_gain
            self.is_modified = True
            self.data_changed.emit()
            
            # 如果有插值數據，重新計算
            if self.is_interpolated:
                self.interpolate()
    
    def save_to_file(self, filepath: str, include_interpolated: bool = False) -> bool:
        """儲存數據到檔案"""
        try:
            if include_interpolated and self.is_interpolated:
                frequencies = self.interpolated_frequencies
                gains = self.interpolated_gains
            else:
                frequencies = self.frequencies
                gains = self.gains
            
            # 構造輸出字串
            content = "FilterCurve:"
            
            # 寫入頻率數據
            for i, freq in enumerate(frequencies):
                content += f' f{i}="{freq:.6g}"'
            
            # 寫入增益數據  
            for i, gain in enumerate(gains):
                content += f' v{i}="{gain:.6g}"'
            
            # 寫入元數據
            for key, value in self.metadata.items():
                content += f' {key}="{value}"'
            
            # 寫入檔案
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"成功儲存到: {filepath}")
            return True
            
        except Exception as e:
            print(f"儲存錯誤: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """從檔案載入數據"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if self.parse_content(content):
                import os
                self.filename = os.path.basename(filepath)
                self.is_modified = False
                return True
            return False
            
        except Exception as e:
            print(f"載入檔案錯誤: {e}")
            return False
    
    def get_info_text(self) -> str:
        """獲取檔案資訊文字"""
        if len(self.frequencies) == 0:
            return "尚未載入數據"
        
        info = []
        info.append(f"檔案: {self.filename or '未命名'}")
        info.append(f"數據點數: {len(self.frequencies)}")
        info.append(f"頻率範圍: {self.frequencies[0]:.0f} - {self.frequencies[-1]:.0f} Hz")
        info.append(f"增益範圍: {np.min(self.gains):.2f} - {np.max(self.gains):.2f} dB")
        
        if self.is_interpolated:
            info.append(f"\n插值數據:")
            info.append(f"插值點數: {len(self.interpolated_frequencies)}")
            info.append(f"插值增益範圍: {np.min(self.interpolated_gains):.2f} - {np.max(self.interpolated_gains):.2f} dB")
        
        return "\n".join(info)

    # ================================================================================
    # 新增功能：預設曲線和工具方法
    # ================================================================================
    
    @classmethod
    def create_harman_target(cls):
        """創建 Harman 目標曲線"""
        data = cls()
        # 簡化的 Harman 曲線
        frequencies = [20, 100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000]
        gains = [0, 0, 0, 0, 0, 3, 2, 0, -2, -5, -8, -10]
        
        data.frequencies = np.array(frequencies, dtype=float)
        data.gains = np.array(gains, dtype=float)
        data.metadata['preset'] = 'Harman'
        data.metadata['description'] = 'Harman 國際標準目標曲線'
        data.filename = 'harman_target.txt'
        
        return data
    
    @classmethod
    def create_flat_response(cls):
        """創建平坦響應曲線"""
        data = cls()
        frequencies = [20, 100, 1000, 10000, 20000]
        gains = [0, 0, 0, 0, 0]
        
        data.frequencies = np.array(frequencies, dtype=float)
        data.gains = np.array(gains, dtype=float)
        data.metadata['preset'] = 'Flat'
        data.metadata['description'] = '平坦頻率響應'
        data.filename = 'flat_response.txt'
        
        return data
    
    def copy_from(self, other):
        """從另一個實例複製數據"""
        if not isinstance(other, FrequencyResponseData):
            return False
        
        self.frequencies = other.frequencies.copy()
        self.gains = other.gains.copy()
        self.metadata = other.metadata.copy()
        self.filename = other.filename
        self.is_interpolated = other.is_interpolated
        
        if other.is_interpolated:
            self.interpolated_frequencies = other.interpolated_frequencies.copy()
            self.interpolated_gains = other.interpolated_gains.copy()
        
        self.data_changed.emit()
        return True
    
    def clear_data(self):
        """清空所有數據"""
        self.frequencies = np.array([])
        self.gains = np.array([])
        self.metadata.clear()
        self.interpolated_frequencies = np.array([])
        self.interpolated_gains = np.array([])
        self.is_interpolated = False
        self.filename = ""
        self.is_modified = False
        self.data_changed.emit()

# ================================================================================
# 工具函數
# ================================================================================

def validate_frequency_response_file(filepath: str) -> bool:
    """驗證頻率響應檔案格式"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本格式檢查
        if not content.strip().startswith('FilterCurve:'):
            return False
        
        # 檢查是否包含頻率和增益數據
        if 'f0=' not in content or 'v0=' not in content:
            return False
        
        return True
        
    except Exception:
        return False

def create_sample_data() -> str:
    """創建範例數據"""
    return '''FilterCurve:f0="20" f1="30" f2="40" f3="50" f4="60" f5="70" f6="80" f7="90" f8="100" f9="135" f10="170" f11="200" f12="250" f13="300" f14="400" f15="500" f16="600" f17="700" f18="800" f19="900" f20="1000" f21="2000" f22="3000" f23="4000" f24="5000" f25="6000" f26="7000" f27="8000" f28="9000" f29="10000" f30="12500" f31="15000" f32="17500" f33="20000" v0="-2" v1="-4.333334" v2="-4.1435184" v3="-5.8518524" v4="-5.8518524" v5="-6.041666" v6="-5.2824078" v7="-5.6620369" v8="-5.8518524" v9="-3.9537029" v10="-3.3842583" v11="-1.8657417" v12="-1.2962971" v13="0.22222137" v14="-0.15740776" v15="0.032407761" v16="0.4120369" v17="0.22222137" v18="0.22222137" v19="-0.15740776" v20="0.4120369" v21="5.3472233" v22="5.3472233" v23="6.1064816" v24="6.8657417" v25="6.8657417" v26="5.9166679" v27="0.032407761" v28="-1.2962971" v29="-3.1944447" v30="-2.245369" v31="-0.5370369" v32="0.60185242" v33="2.5" FilterLength="8191"'''