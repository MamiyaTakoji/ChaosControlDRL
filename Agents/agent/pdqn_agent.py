#from xuance.torch.agents import PDQN_Agent
import numpy as np
from argparse import Namespace
from gym import spaces
from xuance.environment.single_agent_env import Gym_Env
from xuance.common import DummyOffPolicyBuffer
from tqdm import tqdm
from copy import deepcopy
import gym
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from gym import spaces
from xuance.environment.single_agent_env import Gym_Env
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import Agent
from xuance.common import DummyOffPolicyBuffer
from torch import nn
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS
from K import KAN,KBasicQnetwork,Basic_KAN
#from pdqnk_policy import PDQNKPolicy
from gym.spaces import Dict, Space
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS, Optional, Union
from policies.representation import hyperBlock
from policies.deterministic import PDQNKPolicy
from learners.pdqn_learner import PDQN_Learner

class PDQN_Agent(Agent):
    def _build_learner(self, config, policy):
        return PDQN_Learner(config,policy)
    def _build_representation(self,input_space: Optional[Space],
                              config: Namespace) -> Module:
        input_representations = dict(
            input_shape=space2shape(input_space),
            hidden_sizes=config.representation_hidden_size if hasattr(config, "representation_hidden_size") else None,
            normalize=NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None,
            initialize=nn.init.orthogonal_,
            activationMLP=ActivationFunctions[config.activationMLP],
            activationKAN=ActivationFunctions[config.activationKAN],
            kernels=config.kernels if hasattr(config, "kernels") else None,
            strides=config.strides if hasattr(config, "strides") else None,
            filters=config.filters if hasattr(config, "filters") else None,
            fc_hidden_sizes=config.fc_hidden_sizes if hasattr(config, "fc_hidden_sizes") else None,
            device=self.device,
            kanConfig = config.representation_kan_config,
            blockType = config.representation_block_type
            )
        representation = hyperBlock(**input_representations)
        return representation
    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env):
        super(PDQN_Agent, self).__init__(config, envs)
        
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = config.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        self.start_noise, self.end_noise = config.start_noise, config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

        self.observation_space = envs.observation_space
        
        self.action_space = envs.action_space
        num_disact = self.action_space[0].n
        self.num_disact = num_disact
        self.action_high = [self.action_space.spaces[i].high for i in range(1, num_disact + 1)]
        self.action_low = [self.action_space.spaces[i].low for i in range(1, num_disact + 1)]
        self.action_range = [self.action_space.spaces[i].high - self.action_space.spaces[i].low for i in
                             range(1, num_disact + 1)]
        self.representation_info_shape = {'state': (envs.observation_space)}
        self.auxiliary_info_shape = {}
        self.epsilon = 1.0
        self.epsilon_steps = 1000
        self.epsilon_initial = 1.0
        self.epsilon_final = 0.1
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())
        self.buffer_action_space = spaces.Box(np.zeros(self.conact_size+1), np.ones(self.conact_size+1), dtype=np.float64)
        self.atari = True if self.config.env_name == "Atari" else False

        # Build policy, optimizer, scheduler.
        self.policy = self._build_policy()

        self.memory = DummyOffPolicyBuffer(observation_space=self.observation_space,
                                           action_space=self.buffer_action_space,
                                           auxiliary_shape=self.auxiliary_info_shape,
                                           n_envs=self.n_envs,
                                           buffer_size=config.buffer_size,
                                           batch_size=config.batch_size)
        self.learner = self._build_learner(self.config, self.policy)

        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())
    
    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activationKAN = ActivationFunctions[self.config.activationKAN]
        activationMLP = ActivationFunctions[self.config.activationMLP]
        activation_action = ActivationFunctions[self.config.activation_action]
        device = self.device
        # build representation.
        #representation = self._Kbuild_representation(self.config.representation, self.config)
        # build policy.
        # if hasattr(self.config, "grid_size"):
        #     grid_size = self.config.grid_size
        # if hasattr(self.config,"spline_order"):
        #     spline_order = self.config.spline_order
        kanConfig = PDQNKPolicy._setKANconfig(self.config.__dict__)
        blockType = PDQNKPolicy._setBlockType(self.config.__dict__)
        if self.config.policy == "PDQNK_Policy":
            policy =PDQNKPolicy(
                observation_space=self.observation_space, 
                action_space=self.action_space,
                #representation=representation,
                kanConfig = kanConfig,
                blockType = blockType,
                conactor_hidden_size = self.config.conactor_hidden_size,
                qnetwork_hidden_size = self.config.qnetwork_hidden_size,
                normalize = normalize_fn, 
                initialize = initializer, 
                activationMLP = activationMLP, 
                activationKAN = activationKAN,
                activation_action = activation_action,
                device = device,
                use_distributed_training=self.distributed_training,
                )
        else:
            raise AttributeError(
                f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy
    
    def action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            con_actions = self.policy.con_action(obs)
            rnd = np.random.rand()
            if rnd < self.e_greedy:
                disaction = np.random.choice(self.num_disact,self.envs.num_envs)
            else:
                q = self.policy.Qeval(obs, con_actions)
                q = q.detach().cpu().data.numpy()
                disaction = np.argmax(q,axis = 1)

        con_actions = con_actions.cpu().data.numpy()
        #offset = np.array([self.conact_sizes[i] for i in range(disaction)], dtype=int).sum()
        offset = []
        for disa in disaction:
            temp = np.array([self.conact_sizes[i] for i in range(disa)], dtype = int).sum()
            offset.append(temp)
        offset = np.array(offset, dtype = int)
        conaction = []
        for i in range(len(offset)):
            temp = con_actions[i][offset[i]:offset[i] + self.conact_sizes[disaction[i]]]
            conaction.append(temp)
        #conaction = con_actions[offset:offset + self.conact_sizes[disaction]]

        return disaction, conaction, con_actions
    
    def train_epochs(self, n_epochs=1):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample()
            train_info = self.learner.update(**samples)
        return train_info
    
    def pad_action(self, disaction, conaction):
        actions = []
        for j in range(len(disaction)):
            con_actions = []
            #对的对的
            conaction[j] = self.action_range[disaction[j]] * (conaction[j] + 1) / 2. + self.action_low[
                disaction[j]]
            for i in self.conact_sizes:
                con_actions.append(np.zeros((i,), dtype = np.float32))
            con_actions = np.array(con_actions)
            con_actions[disaction[j]][:] = conaction[j]
            actions.append((disaction[j],con_actions))
        return actions
    def train(self, train_steps=10000):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            disaction, conaction, con_actions = self.action(obs)
            action = self.pad_action(disaction, conaction)
            # action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[
            #     disaction]
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
    def _update_explore_factor(self):
       if self.e_greedy is not None:
           if self.e_greedy > self.end_greedy:
               self.e_greedy = self.start_greedy - self.current_step * self.delta_egreedy
       elif self.noise_scale is not None:
           if self.noise_scale >= self.end_noise:
               self.noise_scale = self.start_noise - self.current_step * self.delta_noise
       else:
           return