import sys
sys.path.append(r"d:\Paper\PaperCode4Paper3Final\Agents2")
sys.path.append('/root/PaperCode4Paper3Final/Agents2')
from gym.spaces import Box,Discrete,Tuple,MultiDiscrete
import numpy as np
from DuffingOscillator import model
from xuance.environment import RawEnvironment
from agent.pbdq_agent import PBDQ_Agent as _PBDQ_Agent
from common.dataRecorder import DataRecorder
import time
x_mean = -0.1596470081898484
x_std = 0.8521853042421483
x_dot_mean = -0.17046755852494672
x_dot_std = 0.28534082000181243
MaxStep = 1000
actNumbers = 3
actionLim = 0.0094
FinishValue = 1e-4
def DataProgress(x:np.array):
    #X可能是[2, n]的，也可能是[2, ]
    xcopy = np.array(x)
    if len(x.shape) == 2:
        xcopy[0,:] = (x[0,:] - x_mean)/(1.5*x_std)
        xcopy[1,:] = (x[1,:] - x_dot_mean)/(2*x_dot_std)
    else:
        xcopy[0] = (x[0] - x_mean)/(1.5*x_std)
        xcopy[1] = (x[1] - x_dot_mean)/(2*x_dot_std)
    return xcopy
class ChaosControlRLModel(RawEnvironment):
    def __init__(self, env_config):
        super(ChaosControlRLModel, self).__init__()
        self.env_id = env_config.env_id  # The environment id.
        self.observation_space = Box(0, 1, shape=(2, ))  # 上下限是我随便写的.
        self.actNumbers = actNumbers
        self.action_space = Tuple(
            [MultiDiscrete([actNumbers]), Box(-0.008,0.008), Box(-0.008,0.008), Box(-0.008,0.008)])
        self.max_episode_steps = MaxStep  # The max episode length.
        self._current_step = 0  # The count of steps of current episode.
        self.actionLim = actionLim
        self.ChaosControl = ChaosControl(actionLim)
    def reset(self,**kwargs):
        self._current_step = 0
        self.ChaosControl.reset()
        return self.ChaosControl.state,{}
    def isTerminated(self,loss):
        terminated = False
        if abs(loss) <= FinishValue:
            terminated = True
        return terminated
    def step(self,action):
        self._current_step += 1
        temp = action[0]
        u1 = (temp-1)*self.actionLim
        u2 = action[int(temp)+1]*self.actionLim
        u = u1 + u2
        observation, rewards = self.ChaosControl.step(u)
        terminated = self.isTerminated(rewards)
        #terminated = False
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return DataProgress(observation), rewards, terminated, truncated, info
        #return observation, rewards, terminated, truncated, info
    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return np.ones([64, 64, 64])
    def close(self):  # Close your environment.
        return
class ChaosControl:
    def __init__(self,actionLim):
        self.reset()
        self.actionLim = actionLim
        self.model = model()
    def step(self, u):
        state = self.state
        #变成(2, 1)的矩阵分别矩阵乘法
        u = np.clip(u,-self.actionLim,self.actionLim)
        next_state = self.model.Poincare(state, u)
        loss = -(DataProgress(state) - DataProgress(next_state))*\
        (DataProgress(state) - DataProgress(next_state))/2
        loss = np.sum(loss)
        self.state = next_state
        rewards = loss
        return self.state,rewards
    def reset(self):
        #生成(4, )的矩阵
        self.state = 0.4*np.random.random([2,])-0.2
class Tester:
    def step(self,points,u):
        #注意state需要是2*n的，u是长度为n的向量
        state = points
        m = model()
        next_state = m.PoincareMulti(state.copy(), u)
        loss = -(state - next_state)*(state - next_state)/(4*np.pi*np.pi)
        loss = np.sum(loss, axis = 1)
        self.state = next_state
        rewards = loss
        return self.state,rewards
    def __init__(self):
        self.gridSize = 20
        self.actionLim = actionLim
        self.Reset()
    def GetRealAct(self,action):
        action = np.array(action)
        temp = np.array(action[:,0],dtype = int)
        row = np.arange(0,len(temp))
        u1 = (temp-1)*self.actionLim
        u2 = action[row,temp+1]*self.actionLim
        u = u1 + u2
        u = np.clip(u, -self.actionLim, self.actionLim) 
        return u
    def Reset(self):
        xs = np.linspace(-0.5, 0.5,self.gridSize)
        ys = np.linspace(-0.5, 0.5,self.gridSize)
        _X, _Y = np.meshgrid(xs,ys)
        X = _X.reshape([self.gridSize*self.gridSize])
        Y = _Y.reshape([self.gridSize*self.gridSize])
        points = np.zeros([2, self.gridSize*self.gridSize])
        points[0, :] = X; points[1, :] = Y
        self.points = points
    def Test(self,Agent):
        #全部重写太麻烦了，action那里直接转秩吧
        self.Reset()
        Steps = MaxStep*2*np.ones_like(self.points[0,:])
        IsFinish = np.zeros_like(self.points[0,:],dtype=bool)
        for i in range(MaxStep):
            if self.points[:,(~IsFinish)].shape[1]!=0:
                disaction, con_actions = Agent.action(DataProgress(self.points[:,(~IsFinish)]).T,test_mode = True)
                action = Agent.pad_action(disaction, con_actions)
                realact = self.GetRealAct(action)
                next_points = np.array(self.points)
                #只更新没有失败并且没有完成的点
                next_points[:,(~IsFinish)], _ = self.step(self.points[:,(~IsFinish)], realact)
                #计算成功控制的点
                loss = (DataProgress(next_points) - DataProgress(self.points))**2/2
                loss = np.sum(loss, axis = 0)
                IsFinish = (loss<=FinishValue)
                #更新没有成功控制的点
                Steps[(IsFinish)&(Steps==MaxStep*2)] = i+1
                self.points = next_points
        Steps[Steps == MaxStep*2] = MaxStep
        average_step = np.sum(Steps)/(self.gridSize*self.gridSize)
        return average_step
from tqdm import tqdm
import os
savePath = "bestModels"
class PBDQ_Agent(_PBDQ_Agent):
    def setDataRecorder(self,saveName,seed, best_perform = 1e8):
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
                if average_step < self.best_perform:
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