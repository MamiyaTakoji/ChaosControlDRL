import sys
import os
import shutil
import torch
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
#%%把所有记录统一保存到一个和py文件同名的json文件下

import LearnTrueActionEnv
from common.dataRecorder import DataRecorder
dataRecorder = DataRecorder("LearnTrueActionEnv_Train.json")
seedList = [42, 422, 4222, 42222, 422222]
actionLimList = [0.0085*4, 0.0085*2, 0.0085]
yamlFileNameList = ['MLP 1024x1024.yaml', 'MLP 1024x512x2.yaml', 'MLP 512x512x4.yaml', "KAN 64x64, g=5, s=3.yaml"]

for actionLim in actionLimList:
    LearnTrueActionEnv.actionLim = actionLim
    LearnTrueActionEnv.SetGmodel()
    for yamlFileName in yamlFileNameList:
        from LearnTrueActionEnv import LearnTrueActionEnvRLModel, DQN_Agent
        logPath = "logs/dqn/MLP"
        #if os.path.exists(logPath) and os.path.isdir(logPath):
        #    shutil.rmtree(logPath)
        configs_dict = get_configs(file_dir=yamlFileName)
        configs = argparse.Namespace(**configs_dict)
        REGISTRY_ENV[configs.env_name] = LearnTrueActionEnvRLModel
        envs = make_envs(configs)
        Key = str((actionLim, yamlFileName))
        bestAcc = 0
        for seed in seedList:
            torch.manual_seed(seed)
            #注意这里要重置智能体
            Agent = DQN_Agent(configs, envs)
            Agent.setDataRecorder(Key, seed, bestAcc = bestAcc)
            Dic, bestAcc = Agent.train(configs.running_steps // configs.parallels)
            #Dic, bestAcc = Agent.train(200, testStep = 20)
            dataRecorder.SaveData(Key,Dic)
            print(f"当前动作限制为{actionLim}，当前Config文件为{yamlFileName}，当前种子为{seed}，当前最高正确率为{bestAcc}")
        dataRecorder.Save()