import os
import torch
import numpy as np
import torch.nn as nn
from xuance.common import Sequence, Optional, Callable, Union, Dict
from copy import deepcopy
from gym.spaces import Space, Discrete, MultiDiscrete
from xuance.torch import Module, Tensor, DistributedDataParallel
from xuance.torch.utils import ModuleType
from policies.core import BasicHyperQhead,ActorHyperNet,CriticHybirdNet,BDQhead,BDQheadV2,BDQheadTest
from common.setKANconfig import setKANconfig
class BasicHyperQnetwork(Module):
    """
    The base class to implement DQN based policy

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        representation (Module): The representation module.
        hidden_size (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 kanConfig: Dict[int, Dict[str, int]],
                 blockType: Sequence[str],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(BasicHyperQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicHyperQhead(
        state_dim  = self.representation.output_shapes['state'][0], 
         n_actions = self.action_dim, 
         hidden_sizes = hidden_size,
         normalize = normalize, 
         initialize = initialize, 
         activationMLP = activationMLP, 
         activationKAN = activationKAN,
         device = device,
         blockType = blockType,
         kanConfig = kanConfig)
        self.target_Qhead = deepcopy(self.eval_Qhead)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.representation._get_name() != "Basic_Identical":
                self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.eval_Qhead = DistributedDataParallel(module=self.eval_Qhead, device_ids=[self.rank])

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The hidden state output by the representation.
            argmax_action: The greedy actions.
            evalQ: The evaluated Q-values.
        """
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values via target networks.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs_target: The hidden state output by the representation.
            argmax_action: The greedy actions from target networks.
            targetQ: The evaluated Q-values output by target Q-network.
        """
        outputs_target = self.target_representation(observation)
        targetQ = self.target_Qhead(outputs_target['state'])
        argmax_action = targetQ.argmax(dim=-1)
        return outputs_target, argmax_action.detach(), targetQ.detach()

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
#representation,不需要了

class PDQNKPolicy(Module):
    @staticmethod
    def _setKANconfig(config):
        conactor = setKANconfig(
            config['conactor_kan_configId'],
            config['conactor_kan_spline_order_Value'],
            config['conactor_kan_grid_sizeValue']
            )
        qnetwork = setKANconfig(
            config['qnetwork_kan_configId'],
            config['qnetwork_kan_spline_order_Value'],
            config['qnetwork_kan_grid_sizeValue']
            )
        Dic = {
            "conactor": conactor,
            "qnetwork": qnetwork
            }
        return Dic
    def _setBlockType(config):
        conactor = config['conactor_block_type']
        qnetwork = config['qnetwork_block_type']
        Dic = {
            "conactor": conactor,
            "qnetwork": qnetwork
            }
        return Dic
        
        # size = len(blockId)
        # configDic = {}
        # for i in range(size):
        #     configDic[blockId[i]] = {
        #         "grid_size":grid_size[i], 
        #         'spline_order':spline_order[i]
        #     }
        #return Dic
    """
    The policy of parameterised deep Q network.

    Args:
        observation_space: The observation spaces.
        action_space: The action spaces.
        representation (Module): The representation module.
        conactor_hidden_size (Sequence[int]): List of hidden units for actor network.
        qnetwork_hidden_size (Sequence[int]): List of hidden units for q network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 observation_space,
                 action_space: Discrete,
                 #representation: Module,#先跑通代码再考虑representation
                 #实际情况是这个PDQN根本没用到representation
                 kanConfig: Dict[str,Dict[int, Dict[str, int]]],
                 blockType: Dict[str,Sequence[str]],
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(PDQNKPolicy, self).__init__()
        # self.representation = representation
        # #不能用deepcopy,要手动复制参数
        # self.target_representation = type(representation)(representation.layers_hidden, representation.device,
        #                                          representation.grid_size, representation.spline_order,)
        # for tp, ep in zip(self.target_representation.parameters(), self.representation.parameters()):
        #     tp.data.copy_(ep.data)
        # self.representation_info_shape = self.representation.output_shapes
        # self.target_representation = deepcopy(representation)
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, len(self.action_space))])
        self.conact_size = int(self.conact_sizes.sum())
        
        self.qnetwork = BasicHyperQhead(
            state_dim = self.observation_space.shape[0] + self.conact_size,
            n_actions = self.num_disact,
            hidden_sizes = qnetwork_hidden_size,
            normalize = normalize,
            initialize = initialize, 
            activationMLP = activationMLP, 
            activationKAN = activationKAN,
            device = device,
            blockType = blockType["qnetwork"],
            kanConfig = kanConfig["qnetwork"])
        self.target_qnetwork = deepcopy(self.qnetwork)
        
        self.conactor = ActorHyperNet(
            state_dim = self.observation_space.shape[0], 
            action_dim = self.conact_size,
            hidden_sizes = conactor_hidden_size,
            normalize = normalize, 
            initialize = initialize, 
            activationMLP = activationMLP,
            activationKAN = activationKAN,
            activation_action = activation_action, 
            device = device,
            blockType = blockType["conactor"],
            kanConfig = kanConfig["conactor"])
        self.target_conactor = deepcopy(self.conactor)
    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    def Qtarget(self, state, action):
        input_q = torch.cat((state, action), dim=1)
        target_q = self.target_qnetwork(input_q)
        return target_q

    def Qeval(self, state, action):
        input_q = torch.cat((state, action), dim=1)
        eval_q = self.qnetwork(input_q)
        return eval_q

    def Qpolicy(self, state):
        conact = self.conactor(state)
        input_q = torch.cat((state, conact), dim=1)
        policy_q = torch.sum(self.qnetwork(input_q))
        return policy_q

    def soft_update(self, tau=0.005):
        # for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
        #     tp.data.mul_(1 - tau)
        #     tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.conactor.parameters(), self.target_conactor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
class TD3Policy(Module):
    """
    The policy of twin delayed deep deterministic policy gradient.

    Args:
        action_space (Space): The action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): List of hidden units for actor network.
        critic_hidden_size (Sequence[int]): List of hidden units for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """
    def _setKANconfig(config):
        actor = setKANconfig(
            config['actor_kan_configId'],
            config['actor_kan_spline_order_Value'],
            config['actor_kan_grid_sizeValue']
            )
        critic = setKANconfig(
            config['critic_kan_configId'],
            config['critic_kan_spline_order_Value'],
            config['critic_kan_grid_sizeValue']
            )
        representation = setKANconfig(
            config['representation_kan_configId'],
            config['representation_kan_spline_order_Value'],
            config['representation_kan_grid_sizeValue']
            )
        Dic = {
            "actor": actor,
            "critic": critic,
            "representation": representation
            }
        return Dic
    def _setBlockType(config):
        actor = config['actor_block_type']
        critic = config['critic_block_type']
        representation = config['representation_block_type']
        Dic = {
            "actor": actor,
            "critic": critic,
            "representation": representation
            }
        return Dic
    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 kanConfig: Dict[str,Dict[int, Dict[str, int]]],
                 blockType: Dict[str,Sequence[str]],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(TD3Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation_info_shape = representation.output_shapes

        self.actor_representation = representation
        self.critic_A_representation = deepcopy(representation)
        self.critic_B_representation = deepcopy(representation)

        self.target_actor_representation = deepcopy(representation)
        self.target_critic_A_representation = deepcopy(representation)
        self.target_critic_B_representation = deepcopy(representation)

        self.actor = ActorHyperNet(
            state_dim = representation.output_shapes['state'][0], 
            action_dim = self.action_dim, 
            hidden_sizes = actor_hidden_size,
            normalize = normalize, 
            initialize = initialize, 
            activationMLP = activationMLP, 
            activationKAN = activationKAN,
            activation_action = activation_action,  
            device = device,
            blockType = blockType["actor"],
            kanConfig = kanConfig["actor"])
        self.critic_A = CriticHybirdNet(
            input_dim = representation.output_shapes['state'][0] + self.action_dim, 
            hidden_sizes = critic_hidden_size,
            normalize = normalize, 
            initialize = initialize, 
            activationMLP = activationMLP, 
            activationKAN = activationKAN,
            device = device,
            blockType = blockType["critic"],
            kanConfig = kanConfig["critic"])
        self.critic_B = CriticHybirdNet(
            input_dim = representation.output_shapes['state'][0] + self.action_dim, 
            hidden_sizes = critic_hidden_size,
            normalize = normalize, 
            activationMLP = activationMLP, 
            activationKAN = activationKAN,
            device = device,
            blockType = blockType["critic"],
            kanConfig = kanConfig["critic"])
        self.target_actor = deepcopy(self.actor)
        self.target_critic_A = deepcopy(self.critic_A)
        self.target_critic_B = deepcopy(self.critic_B)

        # parameters
        self.actor_parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.critic_A_representation.parameters()) + list(
            self.critic_A.parameters()) + list(self.critic_B_representation.parameters()) + list(
            self.critic_B.parameters())

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.actor_representation._get_name() != "Basic_Identical":
                self.actor_representation = DistributedDataParallel(self.actor_representation, device_ids=[self.rank])
            if self.critic_A_representation._get_name() != "Basic_Identical":
                self.critic_A_representation = DistributedDataParallel(self.critic_A_representation,
                                                                       device_ids=[self.rank])
            if self.critic_B_representation._get_name() != "Basic_Identical":
                self.critic_B_representation = DistributedDataParallel(self.critic_B_representation,
                                                                       device_ids=[self.rank])
            self.actor = DistributedDataParallel(module=self.actor, device_ids=[self.rank])
            self.critic_A = DistributedDataParallel(module=self.critic_A, device_ids=[self.rank])
            self.critic_B = DistributedDataParallel(module=self.critic_B, device_ids=[self.rank])

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the actor representations, and the actions.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The output of the actor representations.
            act: The actions calculated by the actor.
        """
        outputs = self.actor_representation(observation)
        act = self.actor(outputs['state'])
        return outputs, act

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values via target networks."""
        outputs_actor = self.target_actor_representation(observation)
        outputs_critic_A = self.target_critic_A_representation(observation)
        outputs_critic_B = self.target_critic_B_representation(observation)
        act = self.target_actor(outputs_actor['state'])
        noise = (torch.randn_like(act) * 0.2).clamp(-0.5, 0.5)
        act = (act + noise).clamp(-1, 1)

        qa = self.target_critic_A(torch.concat([outputs_critic_A['state'], act], dim=-1))
        qb = self.target_critic_B(torch.concat([outputs_critic_B['state'], act], dim=-1))
        min_q = torch.min(qa, qb)
        return min_q[:, 0]

    def Qaction(self, observation: Union[np.ndarray, dict], action: Tensor):
        """Returns the evaluated Q-values of state-action pairs."""
        outputs_critic_A = self.critic_A_representation(observation)
        outputs_critic_B = self.critic_B_representation(observation)
        q_eval_a = self.critic_A(torch.concat([outputs_critic_A['state'], action], dim=-1))
        q_eval_b = self.critic_B(torch.concat([outputs_critic_B['state'], action], dim=-1))
        return q_eval_a[:, 0], q_eval_b[:, 0]

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values by calculating actions via actor networks."""
        outputs_actor = self.actor_representation(observation)
        outputs_critic_A = self.critic_A_representation(observation)
        outputs_critic_B = self.critic_B_representation(observation)
        act = self.actor(outputs_actor['state'])
        q_eval_a = self.critic_A(torch.concat([outputs_critic_A['state'], act], dim=-1)).unsqueeze(dim=1)
        q_eval_b = self.critic_B(torch.concat([outputs_critic_B['state'], act], dim=-1)).unsqueeze(dim=1)
        return (q_eval_a + q_eval_b) / 2.0

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_representation.parameters(), self.target_actor_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_A_representation.parameters(), self.target_critic_A_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_A.parameters(), self.target_critic_A.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_B_representation.parameters(), self.target_critic_B_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_B.parameters(), self.target_critic_B.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
class BDQPolicy(Module):
    def __init__(self,
                 action_space:MultiDiscrete,
                 representation: Module,
                 actionValueNet_hidden_sizes: Sequence[int],
                 stateValueNet_hidden_sizes: Sequence[int],
                 kanConfig: Dict[str,Dict[int, Dict[str, int]]],
                 blockType: Dict[str,Sequence[str]],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(BDQPolicy,self).__init__()
        #初始化BDQhead
        self.total_actions = action_space.nvec[0]*len(action_space.nvec)
        self.num_branches = len(action_space.nvec)
        self.action_dim = action_space.nvec
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.target_representation = deepcopy(representation)
        self.eval_BDQhead = BDQheadV2(
            state_dim = self.representation.output_shapes['state'][0],
            actionValueNet_hidden_sizes = actionValueNet_hidden_sizes,
            stateValueNet_hidden_sizes = stateValueNet_hidden_sizes,
            total_actions = self.total_actions,
            num_branches = self.num_branches,
            kanConfig = kanConfig,
            blockType = blockType,
            normalize = normalize,
            initialize = initialize,
            activationMLP = activationMLP,
            activationKAN = activationKAN,
            device = device
            )
        self.target_BDQhead = deepcopy(self.eval_BDQhead)
    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        evalQ = self.eval_BDQhead(outputs['state'])
        argmax_actions = []
        #evalQ是一个Tensor的list，list的长度为num_branches，
        # 其中的每一个元素代表那个动作的每一个分动作的价值
        #需要返回 outputs, 最优动作, evalQ
        for q in evalQ:
            argmax_action = q.argmax(axis = 1)
            argmax_actions.append(argmax_action)
        return outputs, argmax_actions, evalQ
    def target(self, observation: Union[np.ndarray, dict]):
        outputs_target = self.target_representation(observation)
        targetQ = self.target_BDQhead(outputs_target['state'])
        argmax_actions = []
        #evalQ是一个Tensor的list，list的长度为num_branches，
        # 其中的每一个元素代表那个动作的每一个分动作的价值
        #需要返回 outputs, 最优动作, evalQ
        for q in targetQ:
            argmax_action = q.argmax(axis = 1)
            argmax_actions.append(argmax_action)
            argmax_actions_numpy = []
            for argmax_action in argmax_actions:
                argmax_actions_numpy.append(argmax_action.detach())
            targetQ_numpy = []
            for _targetQ in targetQ:
                targetQ_numpy.append(_targetQ.detach())
        return outputs_target, argmax_actions_numpy, targetQ_numpy
        # elif target_version == "mean":
        #     for dim in range(num_action_streams):
        #         selected_a = tf.argmax(selection_q_tp1[dim], axis=1)
        #         selected_q = tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * q_tp1[dim], axis=1) 
        #         masked_selected_q = (1.0 - done_mask_ph) * selected_q
        #         if dim == 0:
        #             mean_next_q_values = masked_selected_q
        #         else:
        #             mean_next_q_values += masked_selected_q 
        #     mean_next_q_values /= num_action_streams
        #     target_q_values = [rew_t_ph + gamma * mean_next_q_values] * num_action_streams
        #TensorFlow代码
        #target_q_vakues见公式6
    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_BDQhead.parameters(), self.target_BDQhead.parameters()):
            tp.data.copy_(ep)
    def soft_update(self, tau = 0.005):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.eval_BDQhead.parameters(), self.target_BDQhead.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
class PBDQPolicy(Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Module,
                 disc_actionValueNet_hidden_sizes: Sequence[int],
                 stateValueNet_hidden_sizes: Sequence[int],
                 conactor_hidden_size: Sequence[int],
                 kanConfig: Dict[str,Dict[int, Dict[str, int]]],
                 blockType: Dict[str,Sequence[str]],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(PBDQPolicy,self).__init__()
        #已完成（大概）
        """
        获取总动作数，注意此时动作空间的类型是
        self.action_space = Tuple(
        (MultiDiscrete([actNumbers], Box(-0.008,0.008), Box(-0.008,0.008), Box(-0.008,0.008))         
        ))类似这样的，所以总的离散动作的个数从Tuple的第一个元素获取
        """
        temp = action_space[0]
        self.total_actions = temp.nvec[0]*len(temp.nvec)
        self.num_branches = len(temp.nvec)
        self.disc_action_dim = temp.nvec#这里是在干嘛？
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.target_representation = deepcopy(representation)
        """
        先建立连续动作的网络，再建立Q网络吧
        但是之前的尝试中，使用DDQN效果反而变差，···还是先
        BDQheadV2试试看吧
        """
        self.observation_space = observation_space
        """
        建立动作网络只需要知道连续动作空间的维度就好了，也就是Tuple的长度-1
        """
        self.conact_size = len(action_space)-1
        self.conactor = ActorHyperNet(
            state_dim = self.observation_space.shape[0], 
            action_dim = self.conact_size,
            hidden_sizes = conactor_hidden_size,
            normalize = normalize, 
            initialize = initialize, 
            activationMLP = activationMLP,
            activationKAN = activationKAN,
            activation_action = activation_action, 
            device = device,
            blockType = blockType["conactor"],
            kanConfig = kanConfig["conactor"])
        self.target_conactor = deepcopy(self.conactor)
        """
        然后建立BDQ网络
        """
        self.eval_BDQhead = BDQheadV2(
            state_dim = 
        self.representation.output_shapes['state'][0],
            actionValueNet_hidden_sizes = disc_actionValueNet_hidden_sizes,
            stateValueNet_hidden_sizes = stateValueNet_hidden_sizes,
            total_actions = self.total_actions,
            num_branches = self.num_branches,
            kanConfig = kanConfig,
            blockType = blockType,
            normalize = normalize,
            initialize = initialize,
            activationMLP = activationMLP,
            activationKAN = activationKAN,
            device = device
            )
        self.target_BDQhead = deepcopy(self.eval_BDQhead)
    @staticmethod
    def _setKANconfig(config):
        conactor = setKANconfig(
            config['conactor_kan_configId'],
            config['conactor_kan_spline_order_Value'],
            config['conactor_kan_grid_sizeValue']
            )
        qnetwork = setKANconfig(
            config['qnetwork_kan_configId'],
            config['qnetwork_kan_spline_order_Value'],
            config['qnetwork_kan_grid_sizeValue']
            )
        Dic = {
            "conactor": conactor,
            "qnetwork": qnetwork
            }
        return Dic
    def _setBlockType(config):
        conactor = config['conactor_block_type']
        qnetwork = config['qnetwork_block_type']
        Dic = {
            "conactor": conactor,
            "qnetwork": qnetwork
            }
        return Dic
    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact
    
    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction
    
    def Qeval(self, observation, action):
        """
        注意先过representation层再接BDQ层
        """  
        # input_q = torch.cat((state, action), dim=1)
        # temp = self.representation(input_q)
        # eval_q = self.eval_BDQhead(temp)
        # #还要根据q值选择一下动作
        # return eval_q
        input_q = torch.cat((observation, action), dim=1)
        outputs = self.representation(input_q)
        evalQ = self.eval_BDQhead(outputs['state'])
        argmax_actions = []
        #evalQ是一个Tensor的list，list的长度为num_branches，
        # 其中的每一个元素代表那个动作的每一个分动作的价值
        #需要返回 outputs, 最优动作, evalQ
        for q in evalQ:
            argmax_action = q.argmax(axis = 1)
            argmax_actions.append(argmax_action)
        return outputs, argmax_actions, evalQ
    def Qtarget(self, observation, action):
        input_q = torch.cat((observation, action), dim=1)
        outputs = self.target_representation(input_q)
        targetQ = self.target_BDQhead(outputs['state'])
        argmax_actions = []
        for q in targetQ:
            argmax_action = q.argmax(axis = 1)
            argmax_actions.append(argmax_action)
        return outputs, argmax_actions, targetQ
    def Qpolicy(self, state):
        conact = self.conactor(state)
        input_q = torch.cat((state, conact), dim=1)
        temp = self.representation(input_q)['state']
        #这里应该最小化所有动作的Q值的和
        Qs = self.eval_BDQhead(temp)
        policy_q = 0
        for Q in Qs:
            policy_q += torch.sum(Q)
        return policy_q
    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.conactor.parameters(), self.target_conactor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.eval_BDQhead.parameters(), self.target_BDQhead.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        
            
            
            
class PBDQPolicyTest(PBDQPolicy):
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Module,
                 disc_actionValueNet_hidden_sizes: Sequence[int],
                 stateValueNet_hidden_sizes: Sequence[int],
                 conactor_hidden_size: Sequence[int],
                 kanConfig: Dict[str,Dict[int, Dict[str, int]]],
                 blockType: Dict[str,Sequence[str]],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(PBDQPolicyTest, self).__init__(
                    observation_space=observation_space,
                    action_space=action_space,
                    representation=representation,
                    disc_actionValueNet_hidden_sizes=disc_actionValueNet_hidden_sizes,
                    stateValueNet_hidden_sizes=stateValueNet_hidden_sizes,
                    conactor_hidden_size=conactor_hidden_size,
                    kanConfig=kanConfig,
                    blockType=blockType,
                    normalize=normalize,
                    initialize=initialize,
                    activationMLP=activationMLP,
                    activationKAN=activationKAN,
                    activation_action=activation_action,
                    device=device,
                    use_distributed_training=use_distributed_training
                )
        #已完成（大概）
        """
        获取总动作数，注意此时动作空间的类型是
        self.action_space = Tuple(
        (MultiDiscrete([actNumbers], Box(-0.008,0.008), Box(-0.008,0.008), Box(-0.008,0.008))         
        ))类似这样的，所以总的离散动作的个数从Tuple的第一个元素获取
        """
        temp = action_space[0]
        self.total_actions = temp.nvec[0]*len(temp.nvec)
        self.num_branches = len(temp.nvec)
        self.disc_action_dim = temp.nvec#这里是在干嘛？
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.target_representation = deepcopy(representation)
        """
        先建立连续动作的网络，再建立Q网络吧
        但是之前的尝试中，使用DDQN效果反而变差，···还是先
        BDQheadV2试试看吧
        """
        self.observation_space = observation_space
        """
        建立动作网络只需要知道连续动作空间的维度就好了，也就是Tuple的长度-1
        """
        self.conact_size = len(action_space)-1
        self.conactor = ActorHyperNet(
            state_dim = self.observation_space.shape[0], 
            action_dim = self.conact_size,
            hidden_sizes = conactor_hidden_size,
            normalize = normalize, 
            initialize = initialize, 
            activationMLP = activationMLP,
            activationKAN = activationKAN,
            activation_action = activation_action, 
            device = device,
            blockType = blockType["conactor"],
            kanConfig = kanConfig["conactor"])
        self.target_conactor = deepcopy(self.conactor)
        """
        然后建立BDQ网络
        """
        self.eval_BDQhead = BDQheadTest(
            state_dim = 
        self.representation.output_shapes['state'][0],
            actionValueNet_hidden_sizes = disc_actionValueNet_hidden_sizes,
            stateValueNet_hidden_sizes = stateValueNet_hidden_sizes,
            total_actions = self.total_actions,
            num_branches = self.num_branches,
            kanConfig = kanConfig,
            blockType = blockType,
            normalize = normalize,
            initialize = initialize,
            activationMLP = activationMLP,
            activationKAN = activationKAN,
            device = device
            )
        self.target_BDQhead = deepcopy(self.eval_BDQhead)  
def Test():
    state_dim = 4;action_dim = 5
    blockType = ['K','K','K','M']
    kanConfig = {
        0:{"grid_size":5, 'spline_order':3},
        1:{"grid_size":5, 'spline_order':3},
        2:{"grid_size":5, 'spline_order':3},
        3:{"grid_size":5, 'spline_order':3},
        }
    hidden_sizes = [6,7,8]
    from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
    activationMLP = ActivationFunctions["relu"]
    activationKAN = ActivationFunctions['relu']
    device = "cuda:0"
    representation = BasicHyperQhead(state_dim = state_dim,
                        n_actions = action_dim, 
                        blockType = blockType, 
                        kanConfig = kanConfig, 
                        hidden_sizes = hidden_sizes,
                        activationMLP = activationMLP,
                        activationKAN = activationKAN,
                        device = device)
    from gym.spaces import Box,Discrete
    T = BasicHyperQnetwork(
        action_space = Discrete(4),
        representation = representation,
        kanConfig = kanConfig,
        blockType = blockType,
        hidden_size = hidden_sizes,
        activationMLP = activationMLP,
        activationKAN = activationKAN,
        device = device
        )
if __name__ == "__main__":
    Test()































