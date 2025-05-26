import sys
GlobalPath = r"D:\Paper\ChaosControlDRL\Agents2"
sys.path.append(GlobalPath)
sys.path.append('/root/CC3/PaperCode4Paper3Final/Agents2')
from gym.spaces import Box,Discrete,Tuple,MultiDiscrete
import numpy as np
from xuance.environment import RawEnvironment
from agent.pbdq_agent import PBDQ_Agent as _PBDQ_Agent
from common.dataRecorder import DataRecorder
import time
actNumbers = 3
#动作空间的设计要保证充分利用离散动作又有连续动作
#离散动作把0.0085（暂定）等分成若干份
#连续空间在[-0.0085,0.0085]中取值
#这个文件测试Logistics映射的情况
#改这个吧，把[0,1]映射到[-1,1]比较简单
def DataProgress(data):
    new_data = (data-0.5)*2
    return new_data
class ChaosControlRLModel(RawEnvironment):
    def __init__(self, env_config):
        super(ChaosControlRLModel, self).__init__()
        self.env_id = env_config.env_id  # The environment id.
        self.observation_space = Box(0, 1, shape=(1, ))  # Define observation space.
        self.actNumbers = actNumbers
        self.action_space = Tuple(
            [MultiDiscrete([actNumbers]), Box(-0.008,0.008), Box(-0.008,0.008), Box(-0.008,0.008)])
        self.max_episode_steps = 50  # The max episode length.
        self._current_step = 0  # The count of steps of current episode.
        ###
        actionLim = 0.0085
        self.actionLim = actionLim
        self.ChaosControl = ChaosControl(actionLim=actionLim)
    def reset(self,**kwargs):
        self._current_step = 0
        self.ChaosControl.reset()
        return self.ChaosControl.state,{}
    def step(self,action):
        self._current_step += 1
        temp = action[0]
        u1 = (temp-1)*0.008
        #u2 = action[1]*0.008
        u2 = action[int(temp)+1]*0.008
        u = u1 + u2
        u = u
        terminated,rewards = self.ChaosControl.step(u)
        observation = self.ChaosControl.state
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return DataProgress(observation), rewards, terminated, truncated, info
    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return np.ones([64, 64, 64])
    def close(self):  # Close your environment.
        return
class ChaosControl:
    def f(self,x):
        return 3.9*x*(1-x)
    def __init__(self,actionLim):
        self.state = np.random.random(1)[0]
        self.actionLim = actionLim
    def step(self,u):
        IsFinish = False
        state = self.state
        if u>self.actionLim:
            u = self.actionLim
        if u<-self.actionLim:
            u = -self.actionLim
        nextState = self.f(state) + u
        loss = self.state - self.f(state)
        loss = loss*loss
        #loss = max(loss**2,1e-4)

        self.state = nextState
        
        if loss <= 1e-4:
            IsFinish = True
        if self.state>1 or self.state<0:
            rewards = -5
            IsFinish = True
        else:
            rewards = -loss
        return IsFinish,rewards
    def reset(self):
        self.state = np.random.random(1)[0]
class Tester:
    def __init__(self):
        self.pointNum = 100
        self.actionLim = 0.0085
        self.actNumbers = actNumbers
        self.Reset() 
    def GetRealAct(self,action):
        action = np.array(action)
        temp = np.array(action[:,0],dtype = int)
        row = np.arange(0,len(temp))
        u1 = (temp-1)*0.0085
        #u2 = action[1]*0.008
        u2 = action[row,temp+1]*0.0085
        u = u1 + u2
        u = np.clip(u, -self.actionLim, self.actionLim) 
        return u
    def Test(self,Agent):
        self.Reset()
        Steps = 100*np.ones(self.pointNum)
        IsFinish = np.zeros(self.pointNum, dtype = bool)
        IsFail = np.zeros(self.pointNum, dtype = bool)
        for i in range(50):
            disaction, con_actions = Agent.action(DataProgress(self.points),test_mode = True)
            action = Agent.pad_action(disaction, con_actions)
            realact = self.GetRealAct(action)
            next_points = np.array(self.points)
            #只更新没有失败并且没有完成的点
            unFinishPoints = self.points[(~IsFinish)+(~IsFail)]
            next_points[(~IsFinish)+(~IsFail)] = 3.9*unFinishPoints*(1-unFinishPoints)+np.array([realact]).T
            #计算跑出去的点
            IsFail = (next_points>1) + (next_points<0)
            #计算成功控制的点
            loss = (next_points - self.points)**2
            IsFinish = (loss<=1e-4) & ~IsFail
            #更新没有成功控制的点
            Steps[(IsFinish[:,0])&(Steps==100)] = i+1
            IsFinish = IsFinish.reshape([self.pointNum])
            IsFail = IsFail.reshape([self.pointNum])
            self.points = next_points
        #没有成功控制的点记为50
        Steps[Steps == 100] = 50
        average_step = np.sum(Steps)/self.pointNum
        return average_step
    def Reset(self):
        self.points = np.array([np.linspace(0, 1, self.pointNum)]).T
from tqdm import tqdm
savePath = "bestModels"
import os
class PBDQ_Agent(_PBDQ_Agent):
    def setDataRecorder(self,saveName,seed, best_perform = 1e10):
        self.seed = seed
        self.dataRecorder = DataRecorder(saveName)
        self.best_perform = best_perform
    def train(self, train_steps, testStep = 200):
        #预计测试testStep次
        T = Tester()
        N = int(train_steps/testStep)
        #按道理要减去测试时间，不过都要测试，所以应该差不多吧
        startTime = time.time()
        for i in tqdm(range(train_steps)):
            self.one_step_train()
            if i%N == 0:
                currentTime = time.time() - startTime
                average_step = T.Test(self)
                #记录结果,更新最优智能体
                if average_step<self.best_perform:
                    path = os.path.join(savePath, self.dataRecorder.SaveName)
                    self.save_best_model(path,"best_performed_model.pth")
                    self.best_perform = average_step
                step_info = {}
                step_info["average step"] = average_step
                self.log_infos(step_info, self.current_step)
                self.dataRecorder.SaveData(self.seed, (average_step, currentTime))
        self.dataRecorder.UpdataData("BestRecord", self.best_perform)
        return self.dataRecorder.DataDic, self.best_perform
    def save_best_model(self, model_dir_save, model_name):
        if self.distributed_training:
            if self.rank > 0:
                return
        # save the neural networks
        if not os.path.exists(model_dir_save):
            os.makedirs(model_dir_save)
        model_path = os.path.join(model_dir_save, model_name)
        self.learner.save_model(model_path)
        # save the observation status
        if self.use_obsnorm:
            obs_norm_path = os.path.join(model_dir_save, "obs_rms.npy")
            observation_stat = {'count': self.obs_rms.count,
                                'mean': self.obs_rms.mean,
                                'var': self.obs_rms.var}
            np.save(obs_norm_path, observation_stat)
    
    
    
    
    
    
    
    
    
    
    