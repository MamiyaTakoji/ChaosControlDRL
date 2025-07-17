#这个脚本用来比较在OGY以及RL方法下，控制的平均步数
#包括有噪声以及无噪声的情况
import numpy as np
import sys
PerformDic = {
 "OGY":{"Logistic":{}, "Henon":{}, "DoubleRotor":{}, "DuffingOscillator":{}}, 
 "RL":{"Logistic":{}, "Henon":{}, "DoubleRotor":{}, "DuffingOscillator":{}} 
 }
#Logistics和Henon比较简单，DoubleRotor比较麻烦
from HybridMultDimChaosControlEnv import ChaosControlRLModel as LogistcModel
from HybridMultDimChaosControlEnv import PBDQ_Agent as PBDQ_Agent1
from HybridMultDimChaosControlEnv2 import ChaosControlRLModel as HenonModel
from HybridMultDimChaosControlEnv2 import PBDQ_Agent as PBDQ_Agent2
from HybridMultDimChaosControlEnv3 import ChaosControlRLModel as DoubleRotorModel
from HybridMultDimChaosControlEnv3 import PBDQ_Agent as PBDQ_Agent3
sys.path.append(r"d:\Paper\PaperCode4Paper3Final\Agents2")
sys.path.append('/root/CC3/PaperCode4Paper3Final/Agents2')
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from common.dataRecorder import DataRecorder
IsAutoDL = False
dataRecorder = DataRecorder("3ChaosEnvTrain.json")
dataRecorder.Load()
SeedList = [42,422,4222,42222,422222]
YamlFileNames = ["BPDQMLP.yaml","BPDQKAN.yaml","BPDQKAN4Env3.yaml"]
import torch


EnvNames = ['Logistic', 'Henon', 'DoubleRotor' ]
_EnvNames = ['logistic', 'Henon', 'double rotor' ]
EnvDic = {'DoubleRotor':DoubleRotorModel,'Logistic':LogistcModel, 'Henon':HenonModel}
AgentDic = {'DoubleRotor':PBDQ_Agent3,'Logistic':PBDQ_Agent1, 'Henon':PBDQ_Agent2}
YamlDic = {'DoubleRotor': ["BPDQKAN4Env3.yaml","BPDQMLP4Env3.yaml"],
           'Logistic':["BPDQKAN.yaml","BPDQMLP.yaml"],
           'Henon':["BPDQKAN.yaml","BPDQMLP.yaml"]}
AgentFolderDic = {'Logistic':"Agents", 'Henon':"Agents2", "DoubleRotor":"Agents2"}
NoiseRange = 0.2
pointNum = 300*300
import matplotlib.pyplot as plt
#%%Logistic映射的情况
modelPath = r"bestModels\\('Logistic', 'BPDQKAN.yaml')\\best_performed_model.pth"
configs_dict = get_configs("BPDQKAN.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = EnvDic["Logistic"]
envs = make_envs(configs)
Agent = PBDQ_Agent1(configs, envs)
Agent.policy.load_state_dict(torch.load(modelPath))
from HybridMultDimChaosControlEnv import Tester, DataProgress
pointNum = 10000
States = np.linspace(0, 1, pointNum).reshape([pointNum,1])
disaction, con_actions = Agent.action(DataProgress(States),test_mode = True)
action = Agent.pad_action(disaction, con_actions)
T = Tester()
realact = T.GetRealAct(action)
Steps = 100*np.ones(pointNum)
IsFinish = np.zeros(pointNum, dtype = bool)
IsFail = np.zeros(pointNum, dtype = bool)
for i in range(50):
    disaction, con_actions = Agent.action(DataProgress(States),test_mode = True)
    action = Agent.pad_action(disaction, con_actions)
    realact = T.GetRealAct(action)
    nextStates = np.array(States)
    #只更新没有失败并且没有完成的点
    unFinishStates = States[(~IsFinish)+(~IsFail)]
    nextStates[(~IsFinish)+(~IsFail)] = 3.9*unFinishStates*(1-unFinishStates)+np.array([realact]).T
    #计算跑出去的点
    IsFail = (nextStates>1) + (nextStates<0)
    #计算成功控制的点
    loss = (nextStates - States)**2
    IsFinish = (loss<=1e-4) & ~IsFail
    #更新没有成功控制的点
    Steps[(IsFinish[:,0])&(Steps==100)] = i+1
    IsFinish = IsFinish.reshape([pointNum])
    IsFail = IsFail.reshape([pointNum])
    States = nextStates
#没有成功控制的点记为50
Steps[Steps == 100] = 50
PerformDic["RL"]["Logistic"]["normal"] = np.mean(Steps)

#然后考虑噪声的情况，假设噪声为控制输入的NoiseRange倍
States = np.linspace(0, 1, pointNum).reshape([pointNum,1])
T = Tester()
realact = T.GetRealAct(action)
Steps = 100*np.ones(pointNum)
IsFinish = np.zeros(pointNum, dtype = bool)
IsFail = np.zeros(pointNum, dtype = bool)
for i in range(50):
    noise = (np.random.random([pointNum, 1])-0.5)*NoiseRange*T.actionLim
    disaction, con_actions = Agent.action(DataProgress(States),test_mode = True)
    action = Agent.pad_action(disaction, con_actions)
    realact = T.GetRealAct(action)
    nextStates = np.array(States)
    #只更新没有失败并且没有完成的点
    unFinishStates = States[(~IsFinish)+(~IsFail)]
    nextStates[(~IsFinish)+(~IsFail)]\
    = 3.9*unFinishStates*(1-unFinishStates)+np.array([realact]).T+noise[(~IsFinish)+(~IsFail)]
    #计算跑出去的点
    IsFail = (nextStates>1) + (nextStates<0)
    #计算成功控制的点
    loss = (nextStates - States)**2
    IsFinish = (loss<=1e-4) & ~IsFail
    #更新没有成功控制的点
    Steps[(IsFinish[:,0])&(Steps==100)] = i+1
    IsFinish = IsFinish.reshape([pointNum])
    IsFail = IsFail.reshape([pointNum])
    States = nextStates
Steps[Steps == 100] = 50
PerformDic["RL"]["Logistic"]["noise"] = np.mean(Steps)
actionLim = T.actionLim
r = 3.9
#接下来考虑OGY的情况，一维的情况下，直接取A-BK=0，此时K=f'(x^*)
def OGY_Logistic(state):
    #注意state是一个n*1的向量
    K = 1.4
    fixed_point = 1-1/r
    u = np.zeros_like(state)
    _u = (state - fixed_point)*K
    u[np.abs((state - fixed_point)*K)<actionLim] = \
    _u[np.abs((state - fixed_point)*K)<actionLim]
    return u
#先考虑无噪声的情况
MaxStep = 1000
States = np.linspace(0, 1, pointNum).reshape([pointNum,1])
Steps = 2*MaxStep*np.ones(pointNum)
IsFinish = np.zeros(pointNum, dtype = bool)
IsFail = np.zeros(pointNum, dtype = bool)
OGY_uList = np.zeros([MaxStep, pointNum])
for i in range(MaxStep):
    nextStates = np.array(States)
    #只更新没有失败并且没有完成的点
    unFinishStates = States[(~IsFinish)+(~IsFail)]
    u = OGY_Logistic(unFinishStates)
    OGY_uList[i,:] = u[:,0]
    nextStates[(~IsFinish)+(~IsFail)] = \
    3.9*unFinishStates*(1-unFinishStates) \
        + u
    OGY_uList[i,IsFinish] = 0
    #计算跑出去的点
    IsFail = (nextStates>1) + (nextStates<0)
    #计算成功控制的点
    loss = (nextStates - States)**2
    IsFinish = (loss<=1e-4) & ~IsFail
    #更新没有成功控制的点
    Steps[(IsFinish[:,0])&(Steps==2*MaxStep)] = i+1
    IsFinish = IsFinish.reshape([pointNum])
    IsFail = IsFail.reshape([pointNum])
    States = nextStates
#没有成功控制的点记为50
Steps[Steps == 2*MaxStep] = MaxStep
PerformDic["OGY"]["Logistic"]["normal"] = np.mean(Steps)
#再考虑有噪声的情况
MaxStep = 1000
States = np.linspace(0, 1, pointNum).reshape([pointNum,1])
Steps = 2*MaxStep*np.ones(pointNum)
IsFinish = np.zeros(pointNum, dtype = bool)
IsFail = np.zeros(pointNum, dtype = bool)
OGY_uList = np.zeros([MaxStep, pointNum])
for i in range(MaxStep):
    noise = (np.random.random([pointNum, 1])-0.5)*NoiseRange*T.actionLim
    nextStates = np.array(States)
    #只更新没有失败并且没有完成的点
    unFinishStates = States[(~IsFinish)+(~IsFail)]
    u = OGY_Logistic(unFinishStates)
    OGY_uList[i,:] = u[:,0]
    OGY_uList[i,IsFinish] = 0
    nextStates[(~IsFinish)+(~IsFail)] = \
    3.9*unFinishStates*(1-unFinishStates) \
        + u + noise
    #计算跑出去的点
    IsFail = (nextStates>1) + (nextStates<0)
    #计算成功控制的点
    loss = (nextStates - States)**2
    IsFinish = (loss<=1e-4) & ~IsFail
    #更新没有成功控制的点
    Steps[(IsFinish[:,0])&(Steps==2*MaxStep)] = i+1
    IsFinish = IsFinish.reshape([pointNum])
    IsFail = IsFail.reshape([pointNum])
    States = nextStates
#没有成功控制的点记为50
Steps[Steps == 2*MaxStep] = MaxStep
PerformDic["OGY"]["Logistic"]["noise"] = np.mean(Steps)
#%%Henon映射的情况
modelPath = r"bestModels\\('Henon', 'BPDQKAN.yaml')\\best_performed_model.pth"
configs_dict = get_configs("BPDQKAN.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = EnvDic["Henon"]
envs = make_envs(configs)
Agent = PBDQ_Agent2(configs, envs)
Agent.policy.load_state_dict(torch.load(modelPath))
#绘制u1和u2
from HybridMultDimChaosControlEnv2 import Tester
T = Tester()
gridSize = 100
xs = np.linspace(-2,2, gridSize)
ys = np.linspace(-2.5,3, gridSize)
_X, _Y = np.meshgrid(xs,ys)
X = _X.reshape([gridSize*gridSize])
Y = _Y.reshape([gridSize*gridSize])
points = np.array([X,Y]).T
_points = np.array(points)
#获取吸引域
IsOut = np.zeros(gridSize*gridSize,dtype = bool)
for i in range(200):
    NewPoint = np.array(points)
    new_X = 1 - 1.4*points[~IsOut,0]*points[~IsOut,0] + points[~IsOut,1]
    new_Y = 0.3*points[~IsOut,0]
    NewPoint[~IsOut,0] = new_X
    NewPoint[~IsOut,1] = new_Y
    IsOut = np.abs(NewPoint[:,0])>20
    points = np.array(NewPoint)
PArract = np.array(_points)
PArract[IsOut] = np.nan
Point = np.array(PArract)
Steps = 100*np.ones(gridSize*gridSize)
IsFinish = np.zeros(gridSize*gridSize, dtype = bool)
for i in range(50):
    NewPoint = np.array(Point)
    disaction, con_actions = Agent.action(NewPoint,test_mode = True)
    action = Agent.pad_action(disaction, con_actions)
    realact = T.GetRealAct(disaction, con_actions)
    new_X = 1 - 1.4*NewPoint[:,0]*NewPoint[:,0] + NewPoint[:,1]+realact[0]
    new_Y = 0.3*NewPoint[:,0]+realact[1]
    NewPoint[:,0] = new_X
    NewPoint[:,1] = new_Y
    IsFinish = np.sum((NewPoint-Point)**2, axis = 1)<=1e-4
    Steps[np.logical_and(Steps>50,IsFinish)] = i
    Point = np.array(NewPoint)
Steps[Steps==100] = 50
Steps[IsOut] = np.nan
#只统计不是nan的
_Steps = Steps[~IsOut]; Steps = _Steps
PerformDic["RL"]["Henon"]["normal"] = np.mean(Steps)
#接下来考虑有噪声的情况，噪声加在控制项上
Point = np.array(PArract)
Steps = 100*np.ones(gridSize*gridSize)
IsFinish = np.zeros(gridSize*gridSize, dtype = bool)
for i in range(50):
    noise1 = (np.random.random([gridSize*gridSize])-0.5)*0.4*T.actionLim
    noise2 = (np.random.random([gridSize*gridSize])-0.5)*0.4*T.actionLim
    NewPoint = np.array(Point)
    disaction, con_actions = Agent.action(NewPoint,test_mode = True)
    action = Agent.pad_action(disaction, con_actions)
    realact = T.GetRealAct(disaction, con_actions)
    new_X = 1 - 1.4*NewPoint[:,0]*NewPoint[:,0] + NewPoint[:,1] + realact[0]+noise1
    new_Y = 0.3*NewPoint[:,0] + realact[1] + noise2
    NewPoint[:,0] = new_X
    NewPoint[:,1] = new_Y
    IsFinish = np.sum((NewPoint-Point)**2, axis = 1)<=1e-4
    Steps[np.logical_and(Steps>50,IsFinish)] = i
    Point = np.array(NewPoint)
Steps[Steps==100] = 50
Steps[IsOut] = np.nan
_Steps = Steps[~IsOut]; Steps = _Steps
PerformDic["RL"]["Henon"]["noise"] = np.mean(Steps)

#接下来考虑OGY的情况
import control
a = 1.4; b = 0.3
x_F = ((b-1)+np.sqrt((b-1)**2+4*a))/(2*a)
fixed_point = np.array([x_F, b*x_F])
A = np.array([[-2*a*x_F,1],[b,0]])
B = np.diag([1,1])
eig, _ = np.linalg.eig(A)
eig[np.abs(eig)>1] = 0
K = control.place(A, B, eig)
def OGY_Henon(state):
    #输入为n*2的矩阵
    x = state[:,0]; y = state[:,1]
    u = np.zeros_like(state)
    _u = -(state - fixed_point)@K.T
    u[:,0][np.abs(_u[:,0])<actionLim] = _u[:,0][np.abs(_u[:,0])<actionLim]
    u[:,1][np.abs(_u[:,1])<actionLim] = _u[:,1][np.abs(_u[:,1])<actionLim]
    return u
Point = np.array(PArract)
MaxStep = 5000
Steps = 2*MaxStep*np.ones(gridSize*gridSize)
IsFinish = np.zeros(gridSize*gridSize, dtype = bool)
for i in range(MaxStep):
    NewPoint = np.array(Point)
    u = OGY_Henon(NewPoint)
    new_X = 1 - 1.4*NewPoint[:,0]*NewPoint[:,0] + NewPoint[:,1] + u[:,0]
    new_Y = 0.3*NewPoint[:,0] + u[:,1]
    NewPoint[:,0] = new_X
    NewPoint[:,1] = new_Y
    IsFinish = np.sum((NewPoint-Point)**2, axis = 1)<=1e-4
    Steps[np.logical_and(Steps>MaxStep,IsFinish)] = i
    Point = np.array(NewPoint)
Steps[Steps==2*MaxStep] = MaxStep
Steps[IsOut] = np.nan
_Steps = Steps[~IsOut]; Steps = _Steps
PerformDic["OGY"]["Henon"]["normal"] = np.mean(Steps)
#接下来考虑噪声的情况
Point = np.array(PArract)
MaxStep = 5000
Steps = 2*MaxStep*np.ones(gridSize*gridSize)
IsFinish = np.zeros(gridSize*gridSize, dtype = bool)
for i in range(MaxStep):
    noise1 = (np.random.random([gridSize*gridSize])-0.5)*NoiseRange*T.actionLim
    noise2 = (np.random.random([gridSize*gridSize])-0.5)*NoiseRange*T.actionLim
    NewPoint = np.array(Point)
    u = OGY_Henon(NewPoint)
    new_X = 1 - 1.4*NewPoint[:,0]*NewPoint[:,0] + NewPoint[:,1] + u[:,0] + noise1
    new_Y = 0.3*NewPoint[:,0] + u[:,1] + noise2
    NewPoint[:,0] = new_X
    NewPoint[:,1] = new_Y
    IsFinish = np.sum((NewPoint-Point)**2, axis = 1)<=1e-4
    Steps[np.logical_and(Steps>MaxStep,IsFinish)] = i
    Point = np.array(NewPoint)
Steps[Steps==2*MaxStep] = MaxStep
Steps[IsOut] = np.nan
_Steps = Steps[~IsOut]; Steps = _Steps
PerformDic["OGY"]["Henon"]["noise"] = np.mean(Steps)
#%%最麻烦的DoubleRotor
from HybridMultDimChaosControlEnv3 import DataProgress, Tester
modelPath = r"bestModels\\('DoubleRotor', 'BPDQKAN4Env3.yaml')\\best_performed_model.pth"
configs_dict = get_configs("BPDQKAN4Env3.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = EnvDic["DoubleRotor"]
envs = make_envs(configs)
Agent = PBDQ_Agent3(configs, envs)
Agent.policy.load_state_dict(torch.load(modelPath))
gridSize = 100
xs = np.linspace(0,2*np.pi,gridSize,gridSize)
ys = np.linspace(0,2*np.pi,gridSize,gridSize)
_X, _Y = np.meshgrid(xs,ys)
X = _X.reshape([gridSize*gridSize])
Y = _Y.reshape([gridSize*gridSize])
points = np.zeros([4, gridSize*gridSize])
points[0, :] = X; points[1, :] = Y
T = Tester()
T.gridSize = 100
MaxStep = 500
FinishValue = 1e-4
T.Reset()
T.points = points
Steps = (MaxStep+1)*np.ones_like(T.points[0,:])
IsFinish = np.zeros_like(T.points[0,:],dtype=bool)
StepShow = [100,300,499]
for i in range(MaxStep):
    if T.points[:,(~IsFinish)].shape[1]!=0:
        disaction, con_actions = Agent.action(DataProgress(T.points[:,]).T,test_mode = True)
        action = Agent.pad_action(disaction, con_actions)
        realact = T.GetRealAct(action)
        next_points = np.array(T.points)
        next_points, _ = T.step(T.points, realact)
        loss = (DataProgress(next_points) - DataProgress(T.points))**2/4
        loss = np.sum(loss, axis = 0)
        IsFinish = (loss<=FinishValue)
        Steps[(IsFinish)&(Steps==(MaxStep+1))] = i+1
        T.points = next_points
PerformDic["RL"]["DoubleRotor"]["normal"] = np.mean(Steps)
#接下来考虑噪音的情况
actionLim = T.actionLim
points = np.zeros([4, gridSize*gridSize])
points[0, :] = X; points[1, :] = Y
T = Tester()
T.gridSize = 100
MaxStep = 500
FinishValue = 1e-4
T.Reset()
T.points = points
Steps = (MaxStep+1)*np.ones_like(T.points[0,:])
IsFinish = np.zeros_like(T.points[0,:],dtype=bool)
StepShow = [100,300,499]
for i in range(MaxStep):
    if T.points[:,(~IsFinish)].shape[1]!=0:
        noise = (np.random.random(T.gridSize**2)-0.5)*actionLim*NoiseRange
        disaction, con_actions = Agent.action(DataProgress(T.points[:,]).T,test_mode = True)
        action = Agent.pad_action(disaction, con_actions)
        realact = T.GetRealAct(action) + noise
        next_points = np.array(T.points)
        next_points, _ = T.step(T.points, realact)
        loss = (DataProgress(next_points) - DataProgress(T.points))**2/4
        loss = np.sum(loss, axis = 0)
        IsFinish = (loss<=FinishValue)
        Steps[(IsFinish)&(Steps==(MaxStep+1))] = i+1
        T.points = next_points
PerformDic["RL"]["DoubleRotor"]["noise"] = np.mean(Steps)
#%%接下来考虑OGY的情况
#这个系统有很多不动点，只考虑论文实验中提到的那几个
#设置参数
import numpy as np
v1= v2 = T=I=m1=m2=l2=1
l1= 1/np.sqrt(2)
f0 = 9
c1 = l1/I; c2 = l2/I
#c1 = f0*l1/I;c2 = f0*l2/I;
delta = np.sqrt(v1**2+4*v2**2)
lambda1 = -1/2*(v1+2*v2+delta)
lambda2 = -1/2*(v1+2*v2-delta)
a = 1/2*(1+v1/delta);
d = 1/2*(1-v1/delta)
b = -v2/delta
W1 = np.array([[a, b],[b, d]])
W2 = np.array([[d, -b],[-b, a]])
L = W1*np.exp(lambda1*T) + W2*np.exp(lambda2*T)
M = W1*(np.exp(lambda1*T)-1)/lambda1 \
    + W2*(np.exp(lambda2*T)-1)/lambda2
A_v = (L-np.diag([1,1]))@np.linalg.inv(M)
fDic = {
        #f01,f02,f0c
        (0, 1):[-2*np.pi*v1*I/l1, 2*np.pi*v1*I/l2, 2*np.pi*v1*I/l1],
        (0, -1):[2*np.pi*v1*I/l1, -2*np.pi*v1*I/l2, 2*np.pi*v1*I/l1]
        }
def Getx(x):
    x = np.mod(x,2*np.pi)
    if x>np.pi:
        return [x, 3*np.pi - x]
    else:
        return [x, np.pi - x]
def GetFixedPoint(N:np.ndarray, q:int):
    #N类似于np.array([[0,1]]).T这种, q取1-4
    #按论文的数据来吧，懒得算了
    #因为sin的值暂时没有复数的情况，所以先这样考虑
    Y = 2*np.pi*np.linalg.inv(M)@N
    f01,f02,_ = fDic[(N[0][0],N[1][0])]
    x1 = Getx(np.arcsin(f01/f0))
    x2 = Getx(np.arcsin(f02/f0))
    if q == 1:
        return np.array([x1[0], x2[0], Y[0][0], Y[1][0]])
    elif q==2:
        return np.array([x1[0], x2[1], Y[0][0], Y[1][0]])
    elif q==3:
        return np.array([x1[1], x2[0], Y[0][0], Y[1][0]])
    elif q==4:
        return np.array([x1[1], x2[1], Y[0][0], Y[1][0]])
    else:
        print("q的值只能为1,2,3,4")
    return 0
N = [np.array([[0,-1]]).T, np.array([[0,1]]).T]
q = [1, 4]
delta = 1
FinishValue = 1e-4
MaxStep = 10000
import control
from HybridMultDimChaosControlEnv3 import Tester, DataProgress
def OGY_DoubleRotor(state, fixed_point, K):
    #注意状态是4*n的
    dist = state - np.array([fixed_point]).T
    _u = -K@dist
    u = np.zeros_like(_u)
    u[np.abs(_u)<delta] = _u[np.abs(_u)<delta]
    u = u[0]
    return u
average_step = MaxStep
for i in range(2):
    fixed_point = GetFixedPoint(N[i], q[i])
    x1, x2, y1, y2 = fixed_point
    H = f0/I*np.diag([l1*np.cos(x1), l2*np.cos(x2)])
    B = np.array([[0, 0, l1/I*np.sin(x1), l2/I*np.sin(x2)]]).T
    temp1 = np.hstack((np.diag([1,1]), M))
    temp2 = np.hstack((H, L + H@M))
    A = np.vstack((temp1, temp2))
    eig_A, _ = np.linalg.eig(A)
    eig_A[np.abs(eig_A)>1] = 0
    K = control.acker(A, B, eig_A)
    # grid_size = 
    states = np.zeros([4,])
    A_ = A-B@K
    eig_A_, _ = np.linalg.eig(A_)
    T = Tester()
    T.gridSize = 30
    T.Reset()
    Steps = MaxStep*2*np.ones_like(T.points[0,:])
    IsFinish = np.zeros_like(T.points[0,:],dtype=bool)

    for i in range(MaxStep):
        if T.points[:,(~IsFinish)].shape[1]!=0:
            u = OGY_DoubleRotor(T.points, fixed_point, K)
            next_points = np.array(T.points)
            next_points, _ = T.step(np.array(T.points), u)
            loss = (DataProgress(next_points) - DataProgress(T.points))**2/4
            loss = np.sum(loss, axis = 0)
            IsFinish = (loss<=FinishValue)
            Steps[(IsFinish)&(Steps==(MaxStep*2))] = i+1
            T.points = next_points
    Steps[Steps == MaxStep*2] = MaxStep
    p = T.points
    average_step = min(np.sum(Steps)/(T.gridSize*T.gridSize),average_step)
PerformDic["OGY"]["DoubleRotor"]["normal"] = average_step
average_step = MaxStep
for i in range(2):
    fixed_point = GetFixedPoint(N[i], q[i])
    x1, x2, y1, y2 = fixed_point
    H = f0/I*np.diag([l1*np.cos(x1), l2*np.cos(x2)])
    B = np.array([[0, 0, l1/I*np.sin(x1), l2/I*np.sin(x2)]]).T
    temp1 = np.hstack((np.diag([1,1]), M))
    temp2 = np.hstack((H, L + H@M))
    A = np.vstack((temp1, temp2))
    eig_A, _ = np.linalg.eig(A)
    eig_A[np.abs(eig_A)>1] = 0
    K = control.acker(A, B, eig_A)
    # grid_size = 
    states = np.zeros([4,])
    A_ = A-B@K
    eig_A_, _ = np.linalg.eig(A_)
    T = Tester()
    T.gridSize = 30
    T.Reset()
    Steps = MaxStep*2*np.ones_like(T.points[0,:])
    IsFinish = np.zeros_like(T.points[0,:],dtype=bool)

    for i in range(MaxStep):
        noise = (np.random.random(T.gridSize**2)-0.5)*delta*NoiseRange
        if T.points[:,(~IsFinish)].shape[1]!=0:
            u = OGY_DoubleRotor(T.points, fixed_point, K)
            next_points = np.array(T.points)
            next_points, _ = T.step(np.array(T.points), u + noise)
            loss = (DataProgress(next_points) - DataProgress(T.points))**2/4
            loss = np.sum(loss, axis = 0)
            IsFinish = (loss<=FinishValue)
            Steps[(IsFinish)&(Steps==(MaxStep*2))] = i+1
            T.points = next_points
    Steps[Steps == MaxStep*2] = MaxStep
    p = T.points
    average_step = min(np.sum(Steps)/(T.gridSize*T.gridSize),average_step)
PerformDic["OGY"]["DoubleRotor"]["noise"] = average_step
#%%绘制DuffingOscillator的情况
#先考虑OGY的情况
from DuffingOscillator import model
from control import acker
from HybridMultDimChaosControlEnv4 import DataProgress
delta = 0.0094
FinishValue = 1e-4
def OGY_DuffingOscillator(state, fixed_point, K):
    #注意状态是2*n的
    dist = state - np.array([fixed_point]).T
    _u = -K@dist
    u = np.zeros_like(_u)
    u[np.abs(_u)<delta] = _u[np.abs(_u)<delta]
    u = u[0]
    return u
def SetPoints(gridsize):
    xs = np.linspace(-0.5, 0.5, gridsize)
    ys = np.linspace(-0.5, 0.5, gridsize)
    _X, _Y = np.meshgrid(xs,ys)
    X = _X.reshape([gridsize*gridsize])
    Y = _Y.reshape([gridsize*gridsize])
    points = np.zeros([2, gridsize*gridsize])
    points[0, :] = X; points[1, :] = Y
    return points
gridsize = 100
points = SetPoints(gridsize)
fixed_point = np.array([0.5079, -0.3535])
A = np.array([[-3.8958, -6.7048], [0.0161, -0.1131]])
B = np.array([[-5.4938, -0.7330]]).T
#计算K
eig,_ = np.linalg.eig(A)
eig[np.abs(eig)>1] = 0
K = acker(A, B, eig)
m = model()
MaxStep = 2000
Steps = MaxStep*2*np.ones_like(points[0,:])
IsFinish = np.zeros_like(points[0,:],dtype=bool)
for i in range(MaxStep):
    if  points[:,(~IsFinish)].shape[1]!=0:
        next_points = np.array(points)
        next_points[:,(~IsFinish)] = \
        m.PoincareMulti(points[:,(~IsFinish)], OGY_DuffingOscillator(points[:,(~IsFinish)], fixed_point, K))
        #计算成功控制的点
        loss = (DataProgress(next_points) - DataProgress(points))**2/2
        loss = np.sum(loss, axis = 0)
        IsFinish = (loss<=FinishValue)
        #更新没有成功控制的点
        Steps[(IsFinish)&(Steps==MaxStep*2)] = i+1
        points = next_points
Steps[Steps == MaxStep*2] = MaxStep
average_step = np.sum(Steps)/(gridsize*gridsize)
PerformDic["OGY"]["DuffingOscillator"]["normal"] = average_step
#考虑有噪声版本
points = SetPoints(gridsize)
Steps = MaxStep*2*np.ones_like(points[0,:])
IsFinish = np.zeros_like(points[0,:],dtype=bool)
for i in range(MaxStep):
    if  points[:,(~IsFinish)].shape[1]!=0:
        noise = (np.random.random(gridsize**2)-0.5)*delta*NoiseRange
        next_points = np.array(points)
        next_points[:,(~IsFinish)] = \
        m.PoincareMulti(points[:,(~IsFinish)], OGY_DuffingOscillator(points[:,(~IsFinish)], fixed_point, K))+ noise[(~IsFinish)]
        #计算成功控制的点
        loss = (DataProgress(next_points) - DataProgress(points))**2/2
        loss = np.sum(loss, axis = 0)
        IsFinish = (loss<=FinishValue)
        #更新没有成功控制的点
        Steps[(IsFinish)&(Steps==MaxStep*2)] = i+1
        points = next_points
Steps[Steps == MaxStep*2] = MaxStep
average_step = np.sum(Steps)/(gridsize*gridsize)
PerformDic["OGY"]["DuffingOscillator"]["noise"] = average_step
#然后考虑RL的情况
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from HybridMultDimChaosControlEnv4 import ChaosControlRLModel, PBDQ_Agent, Tester, DataProgress
import torch
modelPath = r"bestModels\\('DuffingOscillator', 'BPDQKAN4Env4.yaml')\\best_performed_model.pth"
yamlFileName = "BPDQKAN4Env4.yaml"
configs_dict = get_configs(file_dir=yamlFileName)
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = ChaosControlRLModel
envs = make_envs(configs)
Agent = PBDQ_Agent(config=configs, envs=envs)
Agent.policy.load_state_dict(torch.load(modelPath))
T = Tester()
T.Reset()
T.gridSize = 100
T.Reset()
MaxStep = 500
FinishValue = 1e-4
import numpy as np
Steps = MaxStep*2*np.ones_like(T.points[0,:])
IsFinish = np.zeros_like(T.points[0,:],dtype=bool)
for i in range(MaxStep):
    if T.points[:,(~IsFinish)].shape[1]!=0:
        noise = (np.random.random(gridsize**2)-0.5)*delta*NoiseRange
        disaction, con_actions = Agent.action(DataProgress(T.points).T,test_mode = True)
        action = Agent.pad_action(disaction, con_actions)
        realact = T.GetRealAct(action)+noise
        next_points = np.array(T.points)
        #只更新没有失败并且没有完成的点
        next_points, _ = T.step(T.points, realact)
        #计算成功控制的点
        loss = (DataProgress(next_points) - DataProgress(T.points))**2/2
        loss = np.sum(loss, axis = 0)
        IsFinish = (loss<=FinishValue)
        #更新没有成功控制的点
        Steps[(IsFinish)&(Steps==MaxStep*2)] = i+1
        T.points = next_points
Steps[Steps == MaxStep*2] = MaxStep
average_step = np.sum(Steps)/(T.gridSize*T.gridSize)
average_step = np.sum(Steps)/(T.gridSize*T.gridSize)
PerformDic["RL"]["DuffingOscillator"]["noise"] = average_step

T.Reset()
T.gridSize = 20
MaxStep = 500
FinishValue = 1e-4
import numpy as np
Steps = MaxStep*2*np.ones_like(T.points[0,:])
IsFinish = np.zeros_like(T.points[0,:],dtype=bool)
for i in range(MaxStep):
    if T.points[:,(~IsFinish)].shape[1]!=0:
        disaction, con_actions = Agent.action(DataProgress(T.points).T,test_mode = True)
        action = Agent.pad_action(disaction, con_actions)
        realact = T.GetRealAct(action)
        next_points = np.array(T.points)
        #只更新没有失败并且没有完成的点
        next_points, _ = T.step(T.points, realact)
        #计算成功控制的点
        loss = (DataProgress(next_points) - DataProgress(T.points))**2/2
        loss = np.sum(loss, axis = 0)
        IsFinish = (loss<=FinishValue)
        #更新没有成功控制的点
        Steps[(IsFinish)&(Steps==MaxStep*2)] = i+1
        T.points = next_points
Steps[Steps == MaxStep*2] = MaxStep
average_step = np.sum(Steps)/(T.gridSize*T.gridSize)
PerformDic["RL"]["DuffingOscillator"]["normal"] = average_step




















