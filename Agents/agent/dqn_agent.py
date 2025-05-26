#全部重新写过得了
import torch
from argparse import Namespace
from xuance.common import Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OffPolicyAgent
from policies.representation import hyperBlock
import os
import torch
import wandb
import socket
import numpy as np
import torch.distributed as dist
from abc import ABC
from pathlib import Path
from argparse import Namespace
from mpi4py import MPI
from gym.spaces import Dict, Space
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import destroy_process_group
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS, Optional, Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import REGISTRY_Representation, REGISTRY_Learners, Module
from xuance.torch.utils import nn, NormalizeFunctions, ActivationFunctions, init_distributed_mode
from policies.deterministic import BasicHyperQnetwork
from tqdm import tqdm
from copy import deepcopy
from common.setKANconfig import setKANconfig
class DQN_Agent(OffPolicyAgent):
    """The implementation of Deep Q-Networks (DQN) agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def SetKANConfig(self,config):
        representation = setKANconfig(
                config['representation_kan_configId'],
                config['representation_kan_spline_order_Value'],
                config['representation_kan_grid_sizeValue']
                )
        q_hidden = setKANconfig(
                config['q_hidden_kan_configId'],
                config['q_hidden_kan_spline_order_Value'],
                config['q_hidden_kan_grid_sizeValue']
                )
        Dic = {"representation": representation,
               "q_hidden": q_hidden}
        return Dic
    def SetBlockType(self,config):
        representation = config['representation_block_type']
        q_hidden = config['q_block_type']
        Dic = {"representation": representation,
               "q_hidden": q_hidden}
        return Dic
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],):
        super(DQN_Agent, self).__init__(config, envs)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = config.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy)

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.policy)  # build learner
        self.previous_obs = self.envs.buf_obs
    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activationMLP = ActivationFunctions[self.config.activationMLP]
        activationKAN = ActivationFunctions[self.config.activationKAN]
        device = self.device
        self.kanConfig = self.SetKANConfig(self.config.__dict__)
        self.blockType = self.SetBlockType(self.config.__dict__)
        # build representation.
        representation = self._build_representation(self.observation_space, self.config)

        # # build policy.
        if self.config.policy == "BasicHyper_Q_network":
            policy = BasicHyperQnetwork(
                action_space=self.action_space, 
                representation=representation, 
                kanConfig = self.kanConfig["q_hidden"],
                blockType = self.blockType["q_hidden"],
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, 
                initialize=initializer, 
                activationMLP=activationMLP, 
                activationKAN=activationKAN,
                device=device,
                use_distributed_training=self.distributed_training)
        else:
            raise AttributeError(f"{self.config.agent} does not support the policy named {self.config.policy}.")
        return policy
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
            kanConfig = self.kanConfig["representation"],
            blockType = self.blockType["representation"]
            )
        representation = hyperBlock(**input_representations)
        return representation
    def one_step_train(self):
        step_info = {}
        obs = np.array(self.previous_obs)
        self.obs_rms.update(obs)
        obs = self._process_observation(obs)
        policy_out = self.action(obs, test_mode=False)
        acts = policy_out['actions']
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
        self.previous_obs = np.array(obs)
    def train(self, train_steps):
        for j in tqdm(range(train_steps)):
            self.one_step_train()

    def action(self, observations: np.ndarray,
               test_mode: Optional[bool] = False):
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            actions: The actions to be executed.
            values: The evaluated values.
            dists: The policy distributions.
            log_pi: Log of stochastic actions.
        """
        _, actions_output, _ = self.policy(observations)
        if test_mode:
            actions = actions_output.detach().cpu().numpy()
        else:
            actions = self.exploration(actions_output)
        return {"actions": actions}