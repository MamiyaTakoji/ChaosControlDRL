import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Discrete
from xuance.common import Sequence, Optional, Callable, Union, Dict
from xuance.torch import Tensor, Module
from xuance.torch.utils import ModuleType, mlp_block, gru_block, lstm_block
from xuance.torch.utils import CategoricalDistribution, DiagGaussianDistribution, ActivatedDiagGaussianDistribution
from policies.layers import kan_block
from copy import deepcopy
import sys
sys.path.append('D:\Paper\PaperCode4Paper3\Experiments\KAN_DQN')
#允许通过任意顺序连接
class BasicHyperQhead(Module):
    """
    A base class to build Q network and calculate the Q values.

    Args:
        state_dim (int): The input state dimension.
        n_actions (int): The number of discrete actions.
        hidden_sizes: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

        # def __init__(self,
        #              state_dim: int,
        #              n_actions: int,
        #              hidden_sizes: Sequence[int],
        #              normalize: Optional[ModuleType] = None,
        #              initialize: Optional[Callable[..., Tensor]] = None,
        #              activation: Optional[ModuleType] = None,
        #              device: Optional[Union[str, int, torch.device]] = None):
    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 blockType: Sequence[str],
                 kanConfig: Dict[int, Dict[str, int]],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(BasicHyperQhead, self).__init__()
        size = len(blockType)
        condition = True
        for i in range(size):
            if blockType[i] == 'K':
                if i not in kanConfig.keys():
                    condition = False
        condition = condition&(len(blockType) == len(hidden_sizes) + 1)
        assert condition
        #例如state_dim = 5,h = [6,7,8], blockType = [M,K,M,K]
        #那么生成的结果就是
        #5*6的MLP层，6*7的KAN层，7*8的MLP层，8*action_dim的KAN层
        layers_ = []
        input_shape = (state_dim,)
        size = len(blockType)
        for i in range(size-1):
            if blockType[i] == 'M':
                 block, input_shape = mlp_block(input_shape[0], hidden_sizes[i], normalize, activationMLP, initialize, device)
                 layers_.extend(block)
            elif blockType[i] == 'K':
                 block, input_shape = kan_block(input_shape[0], hidden_sizes[i], 
                                                kanConfig[i],
                                                normalize, activationKAN, initialize, device)
                 layers_.extend(block)
            else:
                raise ValueError("只能选取M或者K作为参数")
        if blockType[size-1] == 'M':
                 block, input_shape = mlp_block(input_shape[0], n_actions, None, None, None, device)
                 layers_.extend(block)
        elif blockType[size-1] == 'K':
                 block, input_shape = kan_block(input_shape[0], n_actions, 
                                                kanConfig[size-1],
                                                None, activationKAN, None, device)
                 layers_.extend(block)
        else:
            raise ValueError("只能选取M或者K作为参数")
        self.model = nn.Sequential(*layers_)
    def forward(self, x: Tensor):
        """
        Returns the output of the Q network.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)
class ActorHyperNet(Module):
    """
    The actor network for deterministic policy, which outputs activated continuous actions directly.

    Args:
        state_dim (int): The input state dimension.
        action_dim (int): The dimension of continuous action space.
        hidden_sizes (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 blockType: Sequence[str],
                 kanConfig: Dict[int, Dict[str, int]],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorHyperNet, self).__init__()
        size = len(blockType)
        condition = True
        for i in range(size):
            if blockType[i] == 'K':
                if i not in kanConfig.keys():
                    condition = False
        condition = condition&(len(blockType) == len(hidden_sizes) + 1)
        assert condition
        layers_ = []
        input_shape = (state_dim,)
        size = len(blockType)
        for i in range(size-1):
            if blockType[i] == 'M':
                 block, input_shape = mlp_block(input_shape[0], hidden_sizes[i], normalize, activationMLP, initialize, device)
                 layers_.extend(block)
            elif blockType[i] == 'K':
                 block, input_shape = kan_block(input_shape[0], hidden_sizes[i], 
                                                kanConfig[i],
                                                normalize, activationKAN, initialize, device)
                 layers_.extend(block)
            else:
                raise ValueError("只能选取M或者K作为参数")
        if blockType[size-1] == 'M':
                 block, input_shape = mlp_block(input_shape[0], action_dim, None, activation_action, None, device)
                 layers_.extend(block)
        elif blockType[size-1] == 'K':
                 block, input_shape = kan_block(input_shape[0], action_dim, 
                                                kanConfig[size-1],
                                                None, activationKAN, None, device)
                 block.append(activation_action())
                 layers_.extend(block)
                 
        self.model = nn.Sequential(*layers_)

    def forward(self, x: Tensor):
        """
        Returns the output of the actor.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)
class CriticHybirdNet(Module):
    """
    The critic network that outputs the evaluated values for states (State-Value) or state-action pairs (Q-value).

    Args:
        input_dim (int): The input dimension (dim_state or dim_state + dim_action).
        hidden_sizes (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 input_dim: int,
                 blockType: Sequence[str],
                 kanConfig: Dict[int, Dict[str, int]],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 #activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CriticHybirdNet, self).__init__()
        size = len(blockType)
        condition = True
        for i in range(size):
            if blockType[i] == 'K':
                if i not in kanConfig.keys():
                    condition = False
        condition = condition&(len(blockType) == len(hidden_sizes) + 1)
        assert condition
        layers_ = []
        input_shape = (input_dim,)
        size = len(blockType)
        for i in range(size-1):
            if blockType[i] == 'M':
                 block, input_shape = mlp_block(input_shape[0], hidden_sizes[i], normalize, activationMLP, initialize, device)
                 layers_.extend(block)
            elif blockType[i] == 'K':
                 block, input_shape = kan_block(input_shape[0], hidden_sizes[i], 
                                                kanConfig[i],
                                                normalize, activationKAN, initialize, device)
                 layers_.extend(block)
            else:
                raise ValueError("只能选取M或者K作为参数")
        if blockType[size-1] == 'M':
                 block, input_shape = mlp_block(input_shape[0], 1, None, None, None, device)
                 layers_.extend(block)
        elif blockType[size-1] == 'K':
                 block, input_shape = kan_block(input_shape[0], 1, 
                                                kanConfig[size-1],
                                                None, activationKAN, None, device)
                 #block.append(activation_action())
                 layers_.extend(block)
        #layers_.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers_)

    def forward(self, x: Tensor):
        """
        Returns the output of the Q network.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)

class BDQhead(Module):
    #num_action_branches == 1的时候效果似乎等于DuelDQN，至少要对吧
    #虽然网络的结构很复杂但好像确实只需要一个网络就够了
    def __init__(self,
                 state_dim:int,
                 actionValueNet_hidden_sizes: Sequence[int],
                 stateValueNet_hidden_sizes: Sequence[int],
                 total_actions:int,
                 num_branches:int,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 aggregator = 'reduceLocalMean',):
        super(BDQhead,self).__init__()
        assert num_branches is not None, "必须指定动作分支数量"
        assert total_actions is not None, "必须指定总动作数量"
        assert state_dim is not None, "必须指定输入维度"
        self.state_dim = state_dim
        self.aggregator = aggregator
        self.num_branches = num_branches
        self.actionValueNet = self.ActionValueNet(
         state_dim, 
         actionValueNet_hidden_sizes, 
         total_actions, 
         num_branches,
         normalize,
         initialize,
         activation,
         device)
        self.stateValueNet = self.StateValueNet(
         state_dim, 
         stateValueNet_hidden_sizes, 
         normalize,
         initialize,
         activation,
         device)
    def _dueling_aggregation(self, action_scores, state_values):
        # 根据聚合方法处理优势值
        if self.aggregator == 'reduceLocalMean':
            adjusted_actions = [a - a.mean(dim=1, keepdim=True) for a in action_scores]
        elif self.aggregator == 'reduceGlobalMean':
            global_mean = torch.stack(action_scores).mean(dim=0)
            adjusted_actions = [a - global_mean for a in action_scores]
        elif self.aggregator == 'reduceLocalMax':
            adjusted_actions = [a - a.max(dim=1, keepdim=True)[0] for a in action_scores]
        else:  # naive
            adjusted_actions = action_scores
        
        # 组合状态值和优势值
        q_values = []
        for i in range(self.num_branches):
            q_values.append(state_values + adjusted_actions[i])
        return q_values
    def forward(self,x):
        action_scores = self.actionValueNet(x)
        state_values = self.stateValueNet(x)
        # 返回一个Tensor的list，list的长度为num_branches，
        # 其中的每一个元素代表那个动作的每一个分动作的价值
        return self._dueling_aggregation(action_scores, state_values)
    class ActionValueNet(Module):
        def __init__(self, 
                     state_dim, 
                     hidden_sizes, 
                     total_actions, 
                     num_branches,
                     normalize: Optional[ModuleType] = None,
                     initialize: Optional[Callable[..., Tensor]] = None,
                     activation: Optional[ModuleType] = None,
                     device: Optional[Union[str, int, torch.device]] = None
                     ):
            super(BDQhead.ActionValueNet, self).__init__()
            self.num_branches = num_branches
            self.actions_per_branch = total_actions // num_branches
            #按道理要实现distributed_single_stream的情况，但是之后再说吧
            self.branches = nn.ModuleList()
            for _ in range(num_branches):
                branch_layers = []; input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h,  normalize, activation, initialize, device)
                    branch_layers.extend(mlp)
                branch_layers.extend(mlp_block(input_shape[0], self.actions_per_branch, None, None, None, device)[0])
                self.branches.append(nn.Sequential(*branch_layers))
        def forward(self,x):
            return [branch(x) for branch in self.branches]
    #只实现not independent的情况
    class StateValueNet(Module):
        def __init__(self,state_dim, hidden_sizes,
                      normalize: Optional[ModuleType] = None,
                      initialize: Optional[Callable[..., Tensor]] = None,
                      activation: Optional[ModuleType] = None,
                      device: Optional[Union[str, int, torch.device]] = None):
            super().__init__()
            layers = []
            input_shape = (state_dim,)
            for h in hidden_sizes:
                mlp, input_shape = mlp_block(input_shape[0], h,  normalize, activation, initialize, device)
                layers.extend(mlp) 
            layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return self.model(x)
    # class StateValueNet(Module):
    #     def __init__(self,state_dim, hidden_sizes, num_branches,
    #                   normalize: Optional[ModuleType] = None,
    #                   initialize: Optional[Callable[..., Tensor]] = None,
    #                   activation: Optional[ModuleType] = None,
    #                   device: Optional[Union[str, int, torch.device]] = None):
    #         super().__init__()
    #         self.num_branches = num_branches
    #         self.branches = nn.ModuleList()
    #         for _ in range(num_branches):
    #             layers = []; input_shape = (state_dim,)
    #             for h in hidden_sizes:
    #                 mlp, input_shape = mlp_block(input_shape[0], h,  normalize, activation, initialize, device)
    #                 layers.extend(mlp) 
    #             layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
    #             self.branches.append(nn.Sequential(*layers))

    #     def forward(self, x):
    #         return [branch(x) for branch in self.branches]
#全部重写吧
class BDQheadV2(Module):
    def __init__(self,
                 state_dim:int,
                 actionValueNet_hidden_sizes: Sequence[int],
                 stateValueNet_hidden_sizes: Sequence[int],
                 total_actions:int,
                 num_branches:int,
                 kanConfig: Dict[str,Dict[int, Dict[str, int]]],
                 blockType: Dict[str,Sequence[str]],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 aggregator = 'native',):
        super(BDQheadV2,self).__init__()
        assert num_branches is not None, "必须指定动作分支数量"
        assert total_actions is not None, "必须指定总动作数量"
        assert state_dim is not None, "必须指定输入维度"
        self.state_dim = state_dim
        self.aggregator = aggregator
        self.num_branches = num_branches
        self.actionValueNet = self.ActionValueNet(
         state_dim, 
         actionValueNet_hidden_sizes, 
         total_actions, 
         num_branches,
         blockType['action_value'],
         kanConfig['action_value'],
         normalize,
         initialize,
         activationMLP,
         activationKAN,
         device)
        self.stateValueNet = self.StateValueNet(
         state_dim, 
         stateValueNet_hidden_sizes,
         blockType['state_value'],
         kanConfig['state_value'],
         normalize,
         initialize,
         activationMLP,
         activationKAN,
         device)
    def _dueling_aggregation(self, action_scores, state_values):
        # 根据聚合方法处理优势值
        if self.aggregator == 'reduceLocalMean':
            adjusted_actions = [a - a.mean(dim=1, keepdim=True) for a in action_scores]
        elif self.aggregator == 'reduceGlobalMean':
            global_mean = torch.stack(action_scores).mean(dim=0)
            adjusted_actions = [a - global_mean for a in action_scores]
        elif self.aggregator == 'reduceLocalMax':
            adjusted_actions = [a - a.max(dim=1, keepdim=True)[0] for a in action_scores]
        else:  # naive
            adjusted_actions = action_scores
        
        # 组合状态值和优势值
        q_values = []
        for i in range(self.num_branches):
            q_values.append(state_values + adjusted_actions[i])
        return q_values
    def forward(self,x):
        action_scores = self.actionValueNet(x)
        state_values = self.stateValueNet(x)
        # 返回一个Tensor的list，list的长度为num_branches，
        # 其中的每一个元素代表那个动作的每一个分动作的价值
        return self._dueling_aggregation(action_scores, state_values)
    class ActionValueNet(Module):
        def __init__(self, 
                     state_dim, 
                     hidden_sizes, 
                     total_actions, 
                     num_branches,
                     blockType: Sequence[str],
                     kanConfig: Dict[int, Dict[str, int]],
                     normalize: Optional[ModuleType] = None,
                     initialize: Optional[Callable[..., Tensor]] = None,
                     activationMLP: Optional[ModuleType] = None,
                     activationKAN: Optional[ModuleType] = None,
                     device: Optional[Union[str, int, torch.device]] = None
                     ):
            super(BDQheadV2.ActionValueNet, self).__init__()
            #判断输入是否符合条件
            size = len(blockType)
            condition = True
            for i in range(size):
                if blockType[i] == 'K':
                    if i not in kanConfig.keys():
                        condition = False
            condition = condition&(len(blockType) == len(hidden_sizes) + 1)
            assert condition
            self.num_branches = num_branches
            self.actions_per_branch = total_actions // num_branches
            #按道理要实现distributed_single_stream的情况，但是之后再说吧
            self.branches = nn.ModuleList()
            for _ in range(num_branches):
                branch_layers = []; input_shape = (state_dim,)
                for i in range(size-1):
                    if blockType[i] == 'M':
                        mlp, input_shape = mlp_block(input_shape[0], hidden_sizes[i], normalize, activationMLP, initialize, device)
                        branch_layers.extend(mlp)
                    elif blockType[i] == 'K':
                        kan, input_shape = kan_block(input_shape[0], hidden_sizes[i], 
                                                       kanConfig[i],
                                                       normalize, activationKAN, initialize, device)
                        branch_layers.extend(kan)
                    else:
                        raise ValueError("只能选取M或者K作为参数")
                if blockType[size-1] == 'M':
                        mlp, input_shape = mlp_block(input_shape[0], self.actions_per_branch, None, activationMLP, initialize, device)
                        branch_layers.extend(mlp)
                elif blockType[size-1] == 'K':
                        kan, input_shape = kan_block(input_shape[0], self.actions_per_branch,
                                                       kanConfig[i],
                                                       None, activationKAN, None, device)
                        branch_layers.extend(kan)
                else:
                    raise ValueError("只能选取M或者K作为参数")
                #branch_layers.extend(mlp_block(input_shape[0], self.actions_per_branch, None, None, None, device)[0])
                self.branches.append(nn.Sequential(*branch_layers))
        def forward(self,x):
            return [branch(x) for branch in self.branches]
    class StateValueNet(Module):
        def __init__(self,state_dim, hidden_sizes,
                     blockType: Sequence[str],
                     kanConfig: Dict[int, Dict[str, int]],
                     normalize: Optional[ModuleType] = None,
                     initialize: Optional[Callable[..., Tensor]] = None,
                     activationMLP: Optional[ModuleType] = None,
                     activationKAN: Optional[ModuleType] = None,
                     device: Optional[Union[str, int, torch.device]] = None):
            super().__init__()
            size = len(blockType)
            condition = True
            for i in range(size):
                if blockType[i] == 'K':
                    if i not in kanConfig.keys():
                        condition = False
            condition = condition&(len(blockType) == len(hidden_sizes) + 1)
            assert condition
            layers = []
            input_shape = (state_dim,)
            for i in range(size-1):
                if blockType[i] == 'M':
                        mlp, input_shape = mlp_block(input_shape[0], hidden_sizes[i], normalize, activationMLP, initialize, device)
                        layers.extend(mlp)
                elif blockType[i] == 'K':
                        kan, input_shape = kan_block(input_shape[0], hidden_sizes[i], 
                                                       kanConfig[i],
                                                       normalize, activationKAN, initialize, device)
                        layers.extend(kan)
                else:
                    raise ValueError("只能选取M或者K作为参数") 
            if blockType[size-1] == 'M':
                layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
            elif blockType[size-1] == 'K':
                layers.extend(kan_block(input_shape[0],1,
                                        kanConfig[size-1],
                                        None,activationKAN, None, device)[0])
            else:
                raise ValueError("只能选取M或者K作为参数")
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return self.model(x)
            # for _ in range(num_branches):
            #     branch_layers = []; input_shape = (state_dim,)
            #     for h in hidden_sizes:
            #         mlp, input_shape = mlp_block(input_shape[0], h,  normalize, activation, initialize, device)
            #         branch_layers.extend(mlp)
            #     branch_layers.extend(mlp_block(input_shape[0], self.actions_per_branch, None, None, None, device)[0])
            #     self.branches.append(nn.Sequential(*branch_layers))

class BDQheadTest(BDQheadV2):#V3实现没有duel的网络，在一维的时候，和PDQN完全一致
    def __init__(self,
                 state_dim:int,
                 actionValueNet_hidden_sizes: Sequence[int],
                 stateValueNet_hidden_sizes: Sequence[int],
                 total_actions:int,
                 num_branches:int,
                 kanConfig: Dict[str,Dict[int, Dict[str, int]]],
                 blockType: Dict[str,Sequence[str]],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 aggregator = 'naive',):
        super(BDQheadTest, self).__init__(
            state_dim=state_dim,
            actionValueNet_hidden_sizes=actionValueNet_hidden_sizes,
            stateValueNet_hidden_sizes=stateValueNet_hidden_sizes,
            total_actions=total_actions,
            num_branches=num_branches,
            kanConfig=kanConfig,
            blockType=blockType,
            normalize=normalize,
            initialize=initialize,
            activationMLP=activationMLP,
            activationKAN=activationKAN,
            device=device,
            aggregator=aggregator
        )
        assert num_branches is not None, "必须指定动作分支数量"
        assert total_actions is not None, "必须指定总动作数量"
        assert state_dim is not None, "必须指定输入维度"
        self.state_dim = state_dim
        self.aggregator = aggregator
        self.num_branches = num_branches
        self.actionValueNet = self.ActionValueNet(
         state_dim, 
         actionValueNet_hidden_sizes, 
         total_actions, 
         num_branches,
         blockType['action_value'],
         kanConfig['action_value'],
         normalize,
         initialize,
         activationMLP,
         activationKAN,
         device)
        self.stateValueNet = None
    def _dueling_aggregation(self, action_scores, state_values):
        # 根据聚合方法处理优势值
        if self.aggregator == 'reduceLocalMean':
            adjusted_actions = [a - a.mean(dim=1, keepdim=True) for a in action_scores]
        elif self.aggregator == 'reduceGlobalMean':
            global_mean = torch.stack(action_scores).mean(dim=0)
            adjusted_actions = [a - global_mean for a in action_scores]
        elif self.aggregator == 'reduceLocalMax':
            adjusted_actions = [a - a.max(dim=1, keepdim=True)[0] for a in action_scores]
        else:  # naive
            adjusted_actions = action_scores
        
        # 组合状态值和优势值
        q_values = []
        for i in range(self.num_branches):
            q_values.append(state_values + adjusted_actions[i])
        return q_values
    def forward(self,x):
        action_scores = self.actionValueNet(x)
        #state_values = self.stateValueNet(x)
        # 返回一个Tensor的list，list的长度为num_branches，
        # 其中的每一个元素代表那个动作的每一个分动作的价值
        return self._dueling_aggregation(action_scores, 0)
    









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
    T = BasicHyperQhead(state_dim = state_dim,
                        n_actions = action_dim, 
                        blockType = blockType, 
                        kanConfig = kanConfig, 
                        hidden_sizes = hidden_sizes,
                        activationMLP = activationMLP,
                        activationKAN = activationKAN,
                        device = device)
    TT = deepcopy(T)
    print(TT)
def Test4BDQhead():
    state_dim = 4
    actionValueNet_hidden_sizes = [32,64]
    stateValueNet_hidden_sizes = [32,64]
    total_actions = 10
    num_branches = 2
    T = BDQhead(state_dim, actionValueNet_hidden_sizes, stateValueNet_hidden_sizes, total_actions, num_branches)
    import torch
    x = torch.tensor([[1,2,3,4],[2,3,4,5],[3,4,5,6]],dtype=torch.float)
    y = T.forward(x)
    Actions = []
    #提取最大动作
    for value in y:
        Action = value.argmax(axis = 1)   
        Actions.append(Action)
    return Actions
if __name__ =="__main__":
    import sys
    sys.path.append('D:\Paper\PaperCode4Paper3\Experiments\KAN_DQN')
    Actions = Test4BDQhead()
    print(Actions)

    
    
    
    
    
    
    
    
    