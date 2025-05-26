import torch
from argparse import Namespace
from xuance.common import Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents.policy_gradient.ddpg_agent import DDPG_Agent
from policies.representation import hyperBlock
from policies.deterministic import TD3Policy
from gym.spaces import Dict, Space
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS, Optional, Union
from torch import nn
from tqdm import tqdm
from copy import deepcopy
class TD3_Agent(DDPG_Agent):
    """The implementation of TD3 agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
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
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 tester = None,
                 testTime = 200):
        super(TD3_Agent, self).__init__(config, envs)
        self.testTime = testTime
        self.tester = None

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activationKAN = ActivationFunctions[self.config.activationKAN]
        activationMLP = ActivationFunctions[self.config.activationMLP]
        activation_action = ActivationFunctions[self.config.activation_action]
        device = self.device
        kanConfig = TD3Policy._setKANconfig(self.config.__dict__)
        blockType = TD3Policy._setBlockType(self.config.__dict__)
        # build representations.
        self.config.representation_kan_config = kanConfig["representation"]
        self.config.representation_block_type = blockType["representation"]
        representation = self._build_representation(self.observation_space, self.config)

        # build policy
        if self.config.policy == "TD3_Policy_K":
            policy = TD3Policy(
                action_space=self.action_space, 
                representation=representation,
                kanConfig=kanConfig,
                blockType=blockType,
                actor_hidden_size=self.config.actor_hidden_size, 
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, 
                initialize=initializer, 
                device=device,
                use_distributed_training=self.distributed_training,
                activationMLP = activationMLP, 
                activationKAN = activationKAN,
                activation_action = activation_action)
        else:
            raise AttributeError(f"TD3 currently does not support the policy named {self.config.policy}.")

        return policy
    def train(self, train_steps):
        obs = self.envs.buf_obs
        testTime = self.testTime#在训练过程中预计记录 testTime 次的测试成绩
        N = int(train_steps/testTime)
        for j in tqdm(range(train_steps)):
            step_info = {}
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
            #记录测试结果
            if j%N==0:
                if self.tester is not None:
                    rewards = self.tester.Test()
                    reward_info = {}
                    reward_info["RealReward"] = {"RealReward":rewards}
                    self.log_infos(reward_info, self.current_step)
            self.current_step += self.n_envs
            self._update_explore_factor()