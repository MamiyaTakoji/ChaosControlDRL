import torch
import torch.nn as nn
from xuance.common import Optional, Sequence, Tuple, Type, Union, Callable
import math
import torch.nn.functional as F
import numpy as np
ModuleType = Type[nn.Module]
#记得改回来
grid_range = [-1, 1]
#grid_range = [0, 2*np.pi]
def mlp_block(input_dim: int,
              output_dim: int,
              normalize: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
              device: Optional[Union[str, int, torch.device]] = None) -> Tuple[Sequence[ModuleType], Tuple[int]]:
    block = []
    linear = nn.Linear(input_dim, output_dim, device=device)
    if initialize is not None:
        initialize(linear.weight)
        nn.init.constant_(linear.bias, 0)
    block.append(linear)
    if activation is not None:
        block.append(activation())
    if normalize is not None:
        block.append(normalize(output_dim, device=device))
    return block, (output_dim,)
def kan_block(input_dim: int,
              output_dim: int,
              size: dict = None,
              normalize: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[
                  torch.Tensor], torch.Tensor]] = None,
              device: Optional[Union[str, int, torch.device]] = None) -> Tuple[Sequence[ModuleType], Tuple[int]]:
    block = []
    kanLinear = KANLinear(input_dim, output_dim, device=device,
                          grid_size=size["grid_size"], spline_order=size["spline_order"],base_activation=activation)
    block.append(kanLinear)
    # if activation is not None:
    # block.append(activation())
    return block, (output_dim,)
#%%
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        device,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.Tanh,
        grid_eps=0.02,
        grid_range = grid_range,
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.device = device
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features).to(device))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
            .to(device))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if base_activation is not None:
            self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1,
                               self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        ).to(self.device)  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        x = self.process_input(x, self.device)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        # (in_features, batch_size, out_features)
        B = y.transpose(0, 1).to(self.device)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1).to(self.device)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x):
        x = self.process_input(x, self.device)
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    def process_input(self, x, device):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(device)
        elif isinstance(x, torch.Tensor):
            x = x.to(device)
        else:
            raise ValueError("Unsupported input type")
        return x


























































