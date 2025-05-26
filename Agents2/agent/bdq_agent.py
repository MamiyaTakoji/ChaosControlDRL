import torch
from argparse import Namespace
from xuance.common import Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OffPolicyAgent
from learners.bdq_learner import BDQ_Learner
from policies.deterministic import BDQPolicy
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from policies.representation import hyperBlock
from gym.spaces import Space
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS, Optional, Union
from torch import nn
from common.setKANconfig import setKANconfig
#这个已经不能用了
class BDQ_Agent(OffPolicyAgent):
    def _build_learner(self, config, policy):
        return BDQ_Learner(config,policy)
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
            random_action = np.random.choice(self.action_space.nvec[i],self.n_envs)
            mask = np.random.rand(self.n_envs)<self.e_greedy
            explore_action_numpy[mask] = random_action[mask] 
            explore_actions_numpy.append(explore_action_numpy)
        return explore_actions_numpy
            
        # if self.e_greedy is not None:
        #     random_actions = np.random.choice(self.action_space.n, self.n_envs)
        #     if np.random.rand() < self.e_greedy:
        #         explore_actions = random_actions
        #     else:
        #         explore_actions = pi_actions.detach().cpu().numpy()
        # elif self.noise_scale is not None:
        #     explore_actions = pi_actions + np.random.normal(size=pi_actions.shape) * self.noise_scale
        #     explore_actions = np.clip(explore_actions, self.actions_low, self.actions_high)
        # else:
        #     explore_actions = pi_actions.detach().cpu().numpy()
        return explore_actions
    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
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
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(BDQ_Agent, self).__init__(config, envs)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = config.start_greedy
        #self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy)
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.policy)  # build learner
    
    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "BDQ_Policy":
            policy = BDQPolicy(action_space = self.action_space,
                               representation = representation,
                               actionValueNet_hidden_sizes = self.config.actionValueNet_hidden_sizes,
                               stateValueNet_hidden_sizes = self.config.stateValueNet_hidden_sizes,
                               normalize = normalize_fn,
                               initialize = initializer,
                               activation = activation,
                               device = device)
        else:
            raise AttributeError(f"{self.config.agent} does not support the policy named {self.config.policy}.")

        return policy
class BDQ_AgentV2(OffPolicyAgent):
    def _build_learner(self, config, policy):
        return BDQ_Learner(config,policy)
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
        action_value = setKANconfig(
                config['action_value_kan_configId'],
                config['action_value_kan_spline_order_Value'],
                config['action_value_kan_grid_sizeValue']
                )
        Dic = {"representation": representation,
               "state_value": state_value,
               "action_value":action_value}
        return Dic
    def SetBlockType(self,config):
        representation = config['representation_block_type']
        state_value = config['state_value_block_type']
        action_value = config['action_value_block_type']
        Dic = {"representation": representation,
               "state_value":state_value,
               "action_value":action_value}
        return Dic
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
            #actions = actions_output.detach().cpu().numpy()
            explore_actions_numpy = []
            explore_actions = actions_output
            for i in range(len(explore_actions)):
                explore_action_numpy = explore_actions[i].detach().cpu().numpy()
                explore_actions_numpy.append(explore_action_numpy)
            actions = explore_action_numpy
        else:
            actions = self.exploration(actions_output)
        return {"actions": actions}
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
            random_action = np.random.choice(self.action_space.nvec[i],self.n_envs)
            mask = np.random.rand(self.n_envs)<self.e_greedy
            explore_action_numpy[mask] = random_action[mask] 
            explore_actions_numpy.append(explore_action_numpy)
        return explore_actions_numpy
            
        # if self.e_greedy is not None:
        #     random_actions = np.random.choice(self.action_space.n, self.n_envs)
        #     if np.random.rand() < self.e_greedy:
        #         explore_actions = random_actions
        #     else:
        #         explore_actions = pi_actions.detach().cpu().numpy()
        # elif self.noise_scale is not None:
        #     explore_actions = pi_actions + np.random.normal(size=pi_actions.shape) * self.noise_scale
        #     explore_actions = np.clip(explore_actions, self.actions_low, self.actions_high)
        # else:
        #     explore_actions = pi_actions.detach().cpu().numpy()
        return explore_actions
    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
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
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(BDQ_AgentV2, self).__init__(config, envs)
        self.kanConfig = self.SetKANConfig(config.__dict__)
        self.blockType = self.SetBlockType(config.__dict__)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = config.start_greedy
        #self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy)
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.policy)  # build learner
    
    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activationMLP = ActivationFunctions[self.config.activationMLP]
        activationKAN = ActivationFunctions[self.config.activationKAN]
        
        device = self.device

        # build representation.
        representation = self._build_representation(self.observation_space, self.config)

        # build policy.
        if self.config.policy == "BDQ_Policy":
            policy = BDQPolicy(action_space = self.action_space,
                               representation = representation,
                               actionValueNet_hidden_sizes = self.config.actionValueNet_hidden_sizes,
                               stateValueNet_hidden_sizes = self.config.stateValueNet_hidden_sizes,
                               kanConfig = self.kanConfig,
                               blockType = self.blockType,
                               normalize = normalize_fn,
                               initialize = initializer,
                               activationMLP = activationMLP,
                               activationKAN = activationKAN,
                               device = device)
        else:
            raise AttributeError(f"{self.config.agent} does not support the policy named {self.config.policy}.")

        return policy
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    