import os
import sys
GlobalPath = r"D:\Paper\ChaosControlDRL\Agents"
sys.path.append(GlobalPath)
sys.path.append('/root/CC3/PaperCode4Paper3Final/Agents')
from agent.dqn_agent import DQN_Agent as _DQN_Agent
actionLim = 0.034
stateNum = 10000
actionNum = 5
from xuance.environment import RawEnvironment
from gym.spaces import Box,Discrete
from common.dataRecorder import DataRecorder
#获取Gmodel需要的时间比较长，所以统一在这里生成
from sympy import symbols
Gmodel = None
x_dim = 1; u_dim = 1
X = symbols(f'x:{x_dim}')
U = symbols(f'u:{u_dim}')
r = 3.9
F = [r*X[0]*(1-X[0]) + U[0]]
savePath = "bestModels"
from MyModel2 import MyModel2
def SetGmodel():
    global Gmodel
    Gmodel = MyModel2(F, stateNum)
    Gmodel.SetStateMapingNetWork(actionLim)
class LearnTrueActionEnvRLModel(RawEnvironment):
    def __init__(self, env_config):
        super(LearnTrueActionEnvRLModel, self).__init__()
        self.env_id = env_config.env_id  # The environment id.
        self.observation_space = Box(0, 1, shape=(1, ))  # Define observation space.
        self.action_space = Discrete(6) # Define action space. In this example, the action space is continuous.
        self.max_episode_steps = 100  # The max episode length.
        self._current_step = 0  # The count of steps of current episode.
        self.LearnTrueActionEnv = LearnTrueActionEnv()
    def reset(self):
        self._current_step = 0
        self.LearnTrueActionEnv.reset()
        return self.LearnTrueActionEnv.state,{}
    def step(self,u):
        self._current_step += 1
        observation,rewards = self.LearnTrueActionEnv.step(u)
        truncated = False if self._current_step < self.max_episode_steps else True
        return observation, rewards, False, truncated, {}
    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return np.ones([64, 64, 64])
    def close(self):  # Close your environment.
        return



import numpy as np
class LearnTrueActionEnv:
    def __init__(self):
        self.r = r
        self.state = np.random.random()
        self.FixedPoint = 1-1/r
        self.Gmodel = Gmodel
    def step(self,u):
        #x选取为[0,1]上的数，u选取为0-actionNum之间的数
        _u,_ = self.Gmodel.GetControlItem(np.array([self.state]), np.array([self.FixedPoint]))
        true_action = np.round(actionNum*(_u+actionLim)/(2*actionLim))
        if true_action == u:
            reward = 1
        else:
            reward = 0
        self.state = self.r*self.state*(1-self.state)
        return self.state,reward
    def reset(self):
        self.state = np.random.random()
class Tester:
    def __init__(self):
        self.pointNum = 1000
        self.reset()
        self.Gmodel = Gmodel
        self.FixedPoint = 1-1/r
        if Gmodel is not None:
            true_action, _ = self.Gmodel.GetControlItem(self.points,np.array([self.FixedPoint]))
        self.true_action = np.array(np.round(actionNum*(true_action+actionLim)/(2*actionLim)),dtype = int)
    def reset(self):
        self.points = np.linspace(0, 1, self.pointNum)
    def Test(self,Agent):
        action = Agent.action(self.points.reshape([self.pointNum,1]), test_mode = True)
        acc = np.sum(self.true_action == action['actions'])/self.pointNum
        return acc
from tqdm import tqdm
import time
#主要实现两个功能：
#1.记录准确率的变化
#2.保存效果最好的智能体
class DQN_Agent(_DQN_Agent):
    def setDataRecorder(self,saveName,seed, bestAcc = 0):
        #文件夹的名字形如'(yamlName.yaml,actionLim)'
        self.seed = seed
        self.dataRecorder = DataRecorder(saveName)
        self.bestAcc = bestAcc
    def train(self, train_steps, testStep = 200):
        T = Tester()
        N = int(train_steps/testStep)
        startTime = time.time()
        for i in tqdm(range(train_steps)):
            self.one_step_train()
            if i%N == 0:
                currentTime = time.time() - startTime
                acc = T.Test(self)
                #如果取得了更高的分数，则更新智能体
                if acc>self.bestAcc:
                    path = os.path.join(savePath, self.dataRecorder.SaveName)
                    self.save_best_model(path,"best_performed_model.pth")
                    self.bestAcc = acc
                step_info = {}
                step_info["acc"] = acc
                self.log_infos(step_info, self.current_step)
                self.dataRecorder.SaveData(self.seed, (acc, currentTime))
        self.dataRecorder.UpdataData("BestRecord", self.bestAcc)
        return self.dataRecorder.DataDic, self.bestAcc
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
def Debug():
    T = Tester()
    TA = T.true_action
    return TA
if __name__ == "__main__":
    TA = Debug()













