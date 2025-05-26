from agent.pdqn_agentV1 import PDQN_Agent as _PDQN_Agent
from argparse import Namespace
from xuance.environment.single_agent_env import Gym_Env
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
class PDQN_Agent(_PDQN_Agent):
    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env):
        super(PDQN_Agent, self).__init__(config, envs)
        self.previous_obs = self.envs.buf_obs
    def one_step_train(self):
        step_info = {}
        obs = self.previous_obs
        self.obs_rms.update(obs)
        obs = self._process_observation(obs)
        disaction, con_actions = self.action(obs)
        action = self.pad_action(disaction, con_actions)
        next_obs, rewards, terminals, trunctions, infos = self.envs.step(action)
        disaction = np.array([disaction]).T
        acts = np.concatenate((disaction, con_actions), axis=1)
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
    def train(self,train_steps=10000):
        for _ in tqdm(range(train_steps)):
            self.one_step_train()
    # def action(self, obs, test_mode = False):
    #     with torch.no_grad():
    #         obs = torch.as_tensor(obs, device=self.device).float()
    #         con_actions = self.policy.con_action(obs)
    #         rnd = np.random.rand()
    #         if (rnd < self.e_greedy) and not test_mode:
    #             disaction = np.random.choice(self.num_disact,self.envs.num_envs)
    #         else:
    #             q = self.policy.Qeval(obs, con_actions)
    #             q = q.detach().cpu().data.numpy()
    #             disaction = np.argmax(q,axis = 1)
    
    #     con_actions = con_actions.cpu().data.numpy()
    #     #offset = np.array([self.conact_sizes[i] for i in range(disaction)], dtype=int).sum()
    #     offset = []
    #     for disa in disaction:
    #         temp = np.array([self.conact_sizes[i] for i in range(disa)], dtype = int).sum()
    #         offset.append(temp)
    #     offset = np.array(offset, dtype = int)
    #     conaction = []
    #     for i in range(len(offset)):
    #         temp = con_actions[i][offset[i]:offset[i] + self.conact_sizes[disaction[i]]]
    #         conaction.append(temp)
    #     #conaction = con_actions[offset:offset + self.conact_sizes[disaction]]
    
    #     return disaction, conaction, con_actions   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        