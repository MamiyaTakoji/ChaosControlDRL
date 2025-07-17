import sys
sys.path.append(r"d:\Paper\PaperCode4Paper3Final\Agents2")
sys.path.append('/root/PaperCode4Paper3Final/Agents2')
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from common.dataRecorder import DataRecorder
IsAutoDL = False
dataRecorder = DataRecorder("Duffing Oscillator Model Training Result.json")
SeedList = [42, 422, 4222, 42222, 422222, 0, 1, 2]#Autodl有时候会漏for循环，按道理只跑5个
import torch
import os
import shutil
#%%
from HybirdMultDimChaosControlEnv4 import ChaosControlRLModel as DuffingOscillatorModel
from HybirdMultDimChaosControlEnv4 import PBDQ_Agent
logPath = r"/root/PaperCode4Paper3Final/3addition/logs"
if os.path.exists(logPath) and os.path.isdir(logPath):
    shutil.rmtree(logPath)
YamlList = ["BPDQKAN4Env4.yaml", "BPDQMLP4Env4.yaml",]
for yamlFileName in YamlList:
    configs_dict = get_configs(file_dir=yamlFileName)
    configs = argparse.Namespace(**configs_dict)
    REGISTRY_ENV[configs.env_name] = DuffingOscillatorModel
    envs = make_envs(configs)
    Key = str(("DuffingOscillator", yamlFileName))
    best_perform = 1e10
    count = 0
    for seed in SeedList:
        torch.manual_seed(seed)
        Agent = PBDQ_Agent(config = configs, envs = envs)
        Agent.setDataRecorder(Key, seed, best_perform = best_perform)
        Dic, best_perform = Agent.train(configs.running_steps // configs.parallels)
        dataRecorder.SaveData(Key,Dic)
        print(f"当前Config文件为{yamlFileName}，当前种子为{seed}，当前最优表现为{best_perform}")
        count += 1
        if count==5:
            dataRecorder.Save()
            shutil.rmtree(logPath)
            break


        

















