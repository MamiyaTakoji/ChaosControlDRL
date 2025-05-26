import sys
GlobalPath = r"D:\Paper\ChaosControlDRL\Agents2"
sys.path.append(GlobalPath)
sys.path.append('/root/CC3/PaperCode4Paper3Final/Agents2')
#Henon映射的grid_range应该为[-1,1]
from gym.spaces import Box,Discrete, Tuple,MultiDiscrete
import numpy as np
from xuance.environment import RawEnvironment
from agent.pbdq_agent import PBDQ_Agent as _PBDQ_Agent
from common.dataRecorder import DataRecorder
import time
class ChaosControlRLModel(RawEnvironment):
    def __init__(self, env_config):
        super(ChaosControlRLModel, self).__init__()
        self.env_id = env_config.env_id  # The environment id.
        self.observation_space = Box(0, 1, shape=(2, ))  # Define observation space.
        self.action_space = Tuple(
            (MultiDiscrete([3,3]), 
             Box(-0.008,0.008), Box(-0.008,0.008), Box(-0.008,0.008),
             Box(-0.008,0.008), Box(-0.008,0.008), Box(-0.008,0.008),)         
        )  # Define action space. In this example, the action space is continuous.
        self.max_episode_steps = 50  # The max episode length.
        self._current_step = 0  # The count of steps of current episode.
        actionLim = 0.0085
        self.HenonChaosControl = HenonChaosControl(actionLim=actionLim)
    def reset(self,**kwargs):
        self._current_step = 0
        self.HenonChaosControl.reset()
        return self.HenonChaosControl.state,{}
    def step(self,action):
        self._current_step += 1
        temp = np.array(action[0:2],dtype = int)
        disc_action = (np.array(temp)-1)*0.0085
        con_action1 = action[2:5][temp[0]]*0.0085
        con_action2 = action[5:][temp[1]]*0.0085
        total_action = disc_action + np.array([con_action1,con_action2])
        terminated,rewards = self.HenonChaosControl.step(total_action)
        observation = self.HenonChaosControl.state
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return observation, rewards, terminated, truncated, info
    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return np.ones([64, 64, 64])
    def close(self):  # Close your environment.
        return
class HenonChaosControl:
    def f(self,x):
        #输入二维向量
        x1 = 1 - self.a*x[0]*x[0] + x[1]
        x2 = self.b*x[0]
        return np.array([x1,x2])
    def __init__(self,actionLim):
        #Henon映射的吸引子包含于[-3,3]^2
        self.reset()
        self.actionLim = actionLim
        a = 1.4; self.a = a
        b = 0.3; self.b = b    
    def step(self, real_u):
        IsFinish = False
        state = self.state
        u = np.clip(real_u,
        np.array([-self.actionLim, -self.actionLim]), 
        np.array([self.actionLim, self.actionLim]))
        nextState = self.f(state) + u
        loss = -np.sum((self.state - self.f(state))*(self.state - self.f(state)))
        self.state = nextState
        #如果误差小于1e-4，结束任务
        if abs(loss) <= 1e-4:
            IsFinish = True
            rewards = 0
        #如果离开[-3,3]^2，则结束任务
        elif abs(self.state[0])>3 or abs(self.state[1])>3:
            rewards = -20
            IsFinish = True
        else:
            rewards = loss
        return IsFinish,rewards
    def reset(self):
        self.state = 0.5*(np.random.random(2)-0.5)
class Tester:
    def __init__(self):
        self.gridSize = 15
        self.actionLim = 0.0085
        self.Reset()
    def GetRealAct(self,dis_action,con_action):
        row = np.arange(0, len(dis_action[0]))
        dis_action = np.array(dis_action,dtype = int).T
        u1 = (dis_action[:,0] - 1)*0.0085 \
            + con_action[:,:3][row,dis_action[:,0]]*0.0085
        u2 = (dis_action[:,1] - 1)*0.0085 \
            + con_action[:,3:][row,dis_action[:,1]]*0.0085
        return np.clip(u1, -self.actionLim, self.actionLim),np.clip(u2, -self.actionLim, self.actionLim)
    def Reset(self):
        xs = np.linspace(-0.25,0.25,self.gridSize)
        ys = np.linspace(-0.25,0.25,self.gridSize)
        _X, _Y = np.meshgrid(xs,ys)
        X = _X.reshape([self.gridSize*self.gridSize])
        Y = _Y.reshape([self.gridSize*self.gridSize])
        self.points = np.array([X,Y]).T
    def Test(self,Agent):
        self.Reset()
        Steps = 100*np.ones_like(self.points[:,0])
        IsFinish = np.zeros_like(self.points[:,0],dtype=bool)
        IsFail = np.zeros_like(self.points[:,0],dtype=bool)
        for i in range(50):
            dis_action, con_action = Agent.action(self.points,test_mode = True)
            u1, u2 = self.GetRealAct(dis_action, con_action)
            next_points = np.array(self.points)
            mask = (~IsFinish)+(~IsFail)
            # x1 = 1 - self.a*x[0]*x[0] + x[1]
            # x2 = self.b*x[0]
            new_X = 1 - 1.4*self.points[mask,0]*self.points[mask,0] + self.points[mask,1] + u1
            new_Y = 0.3*self.points[mask,0] + u2
            next_points[mask,0] = new_X
            next_points[mask,1] = new_Y
            #计算跑出去的点
            IsFail = (np.abs(next_points[:,0])>3) + (np.abs(next_points[:,1])>3)
            #计算控制成功的点
            loss = np.sum((self.points-next_points)*(self.points-next_points),axis = 1)
            IsFinish = (loss<=1e-4+1e-5) & ~IsFail
            #更新没有成功控制的点
            Steps[(IsFinish)&(Steps==100)] = i+1
            self.points = next_points
        #没有成功控制的点记为50
        Steps[Steps == 100] = 50
        average_step = np.sum(Steps)/(self.gridSize*self.gridSize)
        return average_step
                
from tqdm import tqdm
savePath = "bestModels"
import os
class PBDQ_Agent(_PBDQ_Agent):
    def setDataRecorder(self,saveName, seed, best_perform = 1e10):
        self.seed = seed
        self.dataRecorder = DataRecorder(saveName)
        self.best_perform = best_perform
    def train(self, train_steps, testStep = 200):
        #预计测试testStep次
        T = Tester()
        N = int(train_steps/testStep)
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





def Debug():
    T = Tester()
    T.Test(None)
if __name__ == "__main__":
    Debug()







































