import numpy as np
from xuance.common import Sequence, Optional, Union, Callable, Dict
from xuance.torch import Module, Tensor
from xuance.torch.utils import torch, nn, mlp_block, ModuleType
from policies.layers import kan_block
class Basic_MLP(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(Basic_MLP, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()
class hyperBlock(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 blockType: Sequence[str],
                 kanConfig: Dict[int, Dict[str, int]],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activationMLP: Optional[ModuleType] = None,
                 activationKAN: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(hyperBlock, self).__init__()
        #方便复制网络
        # self.inputconfig = {
        #     'input_shape': input_shape,
        #     'hidden_sizes': hidden_sizes,
        #     'blockType': blockType,
        #     'kanConfig': kanConfig,
        #     'normalize': normalize,
        #     'initialize': initialize,
        #     'activationMLP': activationMLP,
        #     'activationKAN': activationKAN,
        #     'device': device,
        # }
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initialize = initialize
        self.activationMLP = activationMLP
        self.activationKAN = activationKAN
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        size = len(blockType)
        condition = True
        for i in range(size):
            if blockType[i] == 'K':
                if i not in kanConfig.keys():
                    condition = False
        condition = condition&(len(blockType) == len(hidden_sizes))
        assert condition
        layers_ = []
        input_shape = input_shape
        size = len(blockType)
        for i in range(size):
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
        
        self.model = nn.Sequential(*layers_)
        
    # def _create_network(self):
    #     layers = []
    #     input_shape = self.input_shape
    #     for h in self.hidden_sizes:
    #         mlp, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
    #                                      device=self.device)
    #         layers.extend(mlp)
    #     return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        tensor_observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': self.model(tensor_observation)}