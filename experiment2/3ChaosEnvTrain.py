# -*- coding: utf-8 -*-
"""
Created on Tue May  6 17:23:40 2025

@author: Mamiya
"""
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
IsAutoDL = False

import torch
#%%
from HybridMultDimChaosControlEnv import ChaosControlRLModel as LogistcModel
from HybridMultDimChaosControlEnv import PBDQ_Agent as PBDQ_Agent1
from HybridMultDimChaosControlEnv2 import ChaosControlRLModel as HenonModel
from HybridMultDimChaosControlEnv2 import PBDQ_Agent as PBDQ_Agent2
from HybridMultDimChaosControlEnv3 import ChaosControlRLModel as DoubleRotorModel
from HybridMultDimChaosControlEnv3 import PBDQ_Agent as PBDQ_Agent3
from common.dataRecorder import DataRecorder

dataRecorder = DataRecorder("3ChaosEnvTrain.json")
SeedList = [42,422,4222,42222,422222]
FileNameList = ["BPDQMLP.yaml","BPDQKAN.yaml","BPDQKAN4Env3.yaml"]
EnvNames = ['DoubleRotor','Logistic', 'Henon', ]
EnvDic = {'DoubleRotor':DoubleRotorModel,'Logistic':LogistcModel, 'Henon':HenonModel}
AgentDic = {'DoubleRotor':PBDQ_Agent3,'Logistic':PBDQ_Agent1, 'Henon':PBDQ_Agent2}
YamlDic = {'DoubleRotor': ["BPDQKAN4Env3.yaml","BPDQMLP4Env3.yaml"],
           'Logistic':["BPDQKAN.yaml","BPDQMLP.yaml"],
           'Henon':["BPDQKAN.yaml","BPDQMLP.yaml"]}
AgentFolderDic = {'Logistic':"Agents", 'Henon':"Agents2", "DoubleRotor":"Agents2"}
for envName in EnvNames:
    for yamlFileName in YamlDic[envName]:
        configs_dict = get_configs(file_dir=yamlFileName)
        configs = argparse.Namespace(**configs_dict)
        REGISTRY_ENV[configs.env_name] = EnvDic[envName]
        envs = make_envs(configs)
        Key = str((envName, yamlFileName))
        best_perform = 1e10
        for seed in SeedList:
            torch.manual_seed(seed)
            Agent = AgentDic[envName](config=configs, envs=envs)
            Agent.setDataRecorder(Key, seed, best_perform = best_perform)
            Dic, best_perform = Agent.train(configs.running_steps // configs.parallels)
            dataRecorder.SaveData(Key,Dic)
            print(f"当前环境为{envName}，当前Config文件为{yamlFileName}，当前种子为{seed}，当前最优表现为{best_perform}")
        dataRecorder.Save()



















