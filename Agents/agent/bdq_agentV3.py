#继承bdq_agent，做一些魔改
from agent.bdq_agent import BDQ_AgentV2
from argparse import Namespace
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.common import Union
import numpy as np
from copy import deepcopy
from tqdm import tqdm

class BDQ_AgentV3(BDQ_AgentV2):
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(BDQ_AgentV3, self).__init__(config, envs)
        self.previous_obs = self.envs.buf_obs
    def one_step_train(self):
        step_info = {}
        obs = deepcopy(self.previous_obs)
        self.obs_rms.update(obs)
        obs = self._process_observation(obs)
        policy_out = self.action(obs, test_mode=False)
        acts = policy_out['actions']
        acts = np.array(acts)
        acts = list(acts.T)
        next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

        self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
        if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
            train_info = self.train_epochs(n_epochs=self.n_epochs)
            self.log_infos(train_info, self.current_step)

        self.returns = self.gamma * self.returns + rewards
        obs = deepcopy(next_obs)
        for i in range(self.n_envs):
            if terminals[i] or trunctions[i]:
                if self.atari and (~trunctions[i]):
                    pass
                else:
                    obs[i] = infos[i]["reset_obs"]
                    self.envs.buf_obs[i] = obs[i]
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    self.current_episode[i] += 1
                    if self.use_wandb:
                        step_info[f"Episode-Steps/rank_{self.rank}/env-{i}"] = infos[i]["episode_step"]
                        step_info[f"Train-Episode-Rewards/rank_{self.rank}/env-{i}"] = infos[i]["episode_score"]
                    else:
                        step_info[f"Episode-Steps/rank_{self.rank}"] = {f"env-{i}": infos[i]["episode_step"]}
                        step_info[f"Train-Episode-Rewards/rank_{self.rank}"] = {
                            f"env-{i}": infos[i]["episode_score"]}
                    self.log_infos(step_info, self.current_step)

        self.current_step += self.n_envs
        self._update_explore_factor()
        self.previous_obs = deepcopy(obs)
    def train(self, train_steps):
        for _ in tqdm(range(train_steps)):
            self.one_step_train()
            
        
        
        










































