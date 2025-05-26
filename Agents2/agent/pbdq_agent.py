from xuance.torch.agents import Agent
import numpy as np
from argparse import Namespace
from gym import spaces
from xuance.environment.single_agent_env import Gym_Env
from xuance.common import DummyOffPolicyBuffer
from tqdm import tqdm
from copy import deepcopy
import torch
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.agents import Agent
from torch import nn
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS, Optional 
from gym.spaces import Dict, Space
from policies.representation import hyperBlock
from policies.deterministic import PBDQPolicy, PBDQPolicyTest
from learners.pbdq_learner import PBDQ_Learner
from common.setKANconfig import setKANconfig
from gym.spaces import Box
from xuance.torch.representations import Basic_Identical
class PBDQ_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env):
        super(PBDQ_Agent, self).__init__(config, envs)
        #在config中写入离散和连续动作的维度
        self.config.con_action_dim = len(self.action_space)-1
        self.config.disc_action_dim = len(self.action_space[0].nvec)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = config.start_greedy
        #self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy)
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.policy)  # build learner
        
        self.previous_obs = self.envs.buf_obs
    def _build_learner(self, config, policy):
        return PBDQ_Learner(config, policy)
    def SetKANConfig(self,config):
        representation = setKANconfig(
                config['representation_kan_configId'],
                config['representation_kan_spline_order_Value'],
                config['representation_kan_grid_sizeValue']
                )
        state_value = setKANconfig(
                config['state_value_kan_configId'],
                config['state_value_kan_spline_order_Value'],
                config['state_value_kan_grid_sizeValue']
                )
        disc_action_value = setKANconfig(
                config['disc_action_value_kan_configId'],
                config['disc_action_value_kan_spline_order_Value'],
                config['disc_action_value_kan_grid_sizeValue']
                )
        conactor = setKANconfig(
                config['conactor_kan_configId'],
                config['conactor_kan_spline_order_Value'],
                config['conactor_kan_grid_sizeValue']
                )
        Dic = {"representation": representation,
               "state_value": state_value,
               "action_value":disc_action_value,
               "conactor":conactor}
        return Dic
    def SetBlockType(self,config):
        representation = config['representation_block_type']
        state_value = config['state_value_block_type']
        disc_action_value = config['disc_action_value_block_type']
        conactor = config['conactor_block_type']
        Dic = {"representation": representation,
               "state_value": state_value,
               "action_value": disc_action_value,
               'conactor': conactor}
        return Dic
    def _build_representation(self,input_space: Optional[Space],
                              config: Namespace) -> Module:
        if self.config.representation == 'HybirdBlock':
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
                kanConfig = self.kanConfig["representation"],
                blockType = self.blockType["representation"]
                )
            representation = hyperBlock(**input_representations)
        elif self.config.representation == 'Basic_Identical':
            representation = Basic_Identical(input_shape=space2shape(input_space))
            
        return representation
    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activationKAN = ActivationFunctions[self.config.activationKAN]
        activationMLP = ActivationFunctions[self.config.activationMLP]
        activation_action = ActivationFunctions[self.config.activation_action]
        device = self.device
        self.kanConfig = self.SetKANConfig(self.config.__dict__)
        self.blockType = self.SetBlockType(self.config.__dict__)
        conact_size = len(self.action_space)-1
        input_dim = self.observation_space.shape[0]+conact_size
        input_space = Box(low = np.zeros(input_dim),high = np.ones(input_dim))
        representation = self._build_representation(input_space, self.config)
        if self.config.policy == "PBDQ_Policy":
            policy = PBDQPolicy(
                observation_space = self.observation_space,
                action_space = self.action_space,
                representation = representation,
                disc_actionValueNet_hidden_sizes = 
                self.config.disc_actionValueNet_hidden_sizes,
                stateValueNet_hidden_sizes = 
                self.config.stateValueNet_hidden_sizes,
                conactor_hidden_size = 
                self.config.conactor_hidden_size,
                kanConfig = self.kanConfig,
                blockType = self.blockType,
                normalize = normalize_fn, 
                initialize = initializer, 
                activationMLP = activationMLP, 
                activationKAN = activationKAN,
                activation_action = activation_action,
                device = device,
                use_distributed_training=self.distributed_training,
                )
        elif self.config.policy == "PBDQ_PolicyTest":
            policy = PBDQPolicyTest(
                observation_space = self.observation_space,
                action_space = self.action_space,
                representation = representation,
                disc_actionValueNet_hidden_sizes = 
                self.config.disc_actionValueNet_hidden_sizes,
                stateValueNet_hidden_sizes = 
                self.config.stateValueNet_hidden_sizes,
                conactor_hidden_size = 
                self.config.conactor_hidden_size,
                kanConfig = self.kanConfig,
                blockType = self.blockType,
                normalize = normalize_fn, 
                initialize = initializer, 
                activationMLP = activationMLP, 
                activationKAN = activationKAN,
                activation_action = activation_action,
                device = device,
                use_distributed_training=self.distributed_training,
                )
        return policy
    def action(self, obs, test_mode = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            con_actions = self.policy.con_action(obs)
            _, disc_actions_output, _ = self.policy.Qeval(obs,con_actions)
            if test_mode:
                explore_actions_numpy = []
                explore_actions = disc_actions_output
                for i in range(len(explore_actions)):
                    explore_action_numpy = explore_actions[i].detach().cpu().numpy()
                    explore_actions_numpy.append(explore_action_numpy)
                disc_actions = explore_actions_numpy
            else:
                disc_actions = self.exploration(disc_actions_output)
        #如何组合动作交给环境自己处理
        con_actions = con_actions.cpu().data.numpy()
        return disc_actions,con_actions
    def pad_action(self,disc_actions,con_actions):
        """
        返回组合后的动作
        这个方法只会被用在与环境交互时，所以可以把con_actions转成np.array
        ----------
        disc_actions :
            长度为离散动作维数的列表，列表中的每一个元素的长度为env的数量
        con_actions : 
            大小为环境数x连续动作数的tensor
        Returns
        -------
        actions : 
            长度为环境数的列表
        """
        actions = []
        disc_actions = np.array(disc_actions).T
        actions_numpy = np.hstack((disc_actions,con_actions))
        for j in range(disc_actions.shape[0]):
            actions.append(actions_numpy[j,:])
        return actions
    def exploration(self, pi_actions):
        """Returns the actions for exploration.

        Parameters:
            pi_actions: The original output actions.

        Returns:
            explore_actions: The actions with noisy values.
        """
        explore_actions_numpy = []
        explore_actions = pi_actions
        for i in range(len(explore_actions)):
            explore_action_numpy = explore_actions[i].detach().cpu().numpy()
            random_action = np.random.choice(self.action_space[0].nvec[i],self.n_envs)
            mask = np.random.rand(self.n_envs)<self.e_greedy
            explore_action_numpy[mask] = random_action[mask] 
            explore_actions_numpy.append(explore_action_numpy)
        return explore_actions_numpy
    def train_epochs(self, n_epochs=1):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample()
            train_info = self.learner.update(**samples)
        return train_info
    def train(self, train_steps=10000):
        for _ in tqdm(range(train_steps)):
            self.one_step_train()
    def one_step_train(self):
        obs = deepcopy(self.previous_obs)
        step_info = {}
        disaction, con_actions = self.action(obs)
        action = self.pad_action(disaction, con_actions)
        next_obs, rewards, terminals, trunctions, infos = self.envs.step(action)
        self.memory.store(obs, action, self._process_reward(rewards), terminals, self._process_observation(next_obs))
        if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
            train_info = self.train_epochs(n_epochs=self.n_epochs)
            self.log_infos(train_info, self.current_step)
        
        self.returns = self.gamma * self.returns + rewards
        
        obs = deepcopy(next_obs)
        for i in range(self.n_envs):
            if terminals[i] or trunctions[i]:
                if False and (~trunctions[i]):
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
    def _update_explore_factor(self):
        if self.e_greedy is not None:
            if self.e_greedy > self.end_greedy:
                self.e_greedy = self.start_greedy - self.current_step * self.delta_egreedy
        elif self.noise_scale is not None:
            if self.noise_scale >= self.end_noise:
                self.noise_scale = self.start_noise - self.current_step * self.delta_noise
        else:
            return
        return 
    def _build_memory(self):
        con_action_dim = len(self.action_space)-1
        disc_action_dim = len(self.action_space[0].nvec)
        action_dim = con_action_dim + disc_action_dim
        self.auxiliary_info_shape = {}
        self.buffer_action_space = Box(low = np.zeros(action_dim), high = np.ones(action_dim))
        memory = DummyOffPolicyBuffer(observation_space=self.observation_space,
                                           action_space=self.buffer_action_space,
                                           auxiliary_shape=self.auxiliary_info_shape,
                                           n_envs=self.n_envs,
                                           buffer_size=self.config.buffer_size,
                                           batch_size=self.config.batch_size)
        return memory