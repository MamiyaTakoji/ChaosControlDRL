import json
import os
class DataRecorder:
    def __init__(self, saveName=None):
        self.SaveName = saveName  # 默认保存文件名（可选）
        self.DataDic = {}         # 存储数据的字典
    
    def SaveData(self, Key, Data):
        # 修复逻辑：如果键存在则追加数据，否则创建新键值对
        # 如果Data是字典类型，则合并，如果Datas是列表类型，则增加
        if Key in self.DataDic:
            if type(Data) is dict:
                self.DataDic.update(Data)
            else:
                self.DataDic[Key].append(Data)
        else:
            if type(Data) is dict:
                self.DataDic[Key] = Data
            else:
                self.DataDic[Key] = [Data]
    def UpdataData(self,Key, Data):
        # 覆盖原来的数据
            self.DataDic[Key] = Data
    def Save(self, SaveName=None):
        filename = SaveName if SaveName is not None else self.SaveName
        if not filename:
            raise ValueError("未提供保存文件名")
        
        # 读取原有数据（如果文件存在）
        existing_data = {}
        try:
            with open(filename, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            pass  # 文件不存在时忽略
        
        # 合并新旧数据（与SaveData逻辑一致）
        for key, new_value in self.DataDic.items():
            if key in existing_data:
                existing_value = existing_data[key]
                # 根据类型合并
                if isinstance(existing_value, dict) and isinstance(new_value, dict):
                    existing_value.update(new_value)  # 字典合并
                elif isinstance(existing_value, list) and isinstance(new_value, list):
                    existing_value.extend(new_value)  # 列表追加
                else:
                    # 类型不匹配时覆盖（或自定义其他逻辑）
                    existing_data[key] = new_value
            else:
                existing_data[key] = new_value
        
        # 写入合并后的数据
        with open(filename, "w") as f:
            json.dump(existing_data, f, indent=2)
    
    def Load(self, SaveName=None):
        # 确定最终文件名（同上）
        filename = SaveName if SaveName is not None else self.SaveName
        if not filename:
            raise ValueError("未提供加载文件名")
        
        # 从 JSON 文件读取数据到 DataDic
        with open(filename, "r") as f:
            self.DataDic = json.load(f)