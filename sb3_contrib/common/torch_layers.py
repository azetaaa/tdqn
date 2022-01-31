import math
from typing import Dict, List, Tuple, Type, Union

import torch as th
from torch import nn


def create_noisy_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [NoisyLinear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(NoisyLinear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(NoisyLinear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class NoisyLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise from:
        Fortunato et al. "Noisy Networks for Exploration"
        arXiv:1706.10295v3
    """

    def __init__(self, input_dim, output_dim, sigma=0.3, bias=True):
        super(NoisyLinear, self).__init__(input_dim, output_dim, bias=bias)

        sigma_init = sigma / math.sqrt(input_dim)
        self.sigma_weight = nn.Parameter(th.full((output_dim, input_dim), sigma_init))

        self.register_buffer("epsilon_input", th.zeros(1, input_dim))
        self.register_buffer("epsilon_output", th.zeros(output_dim, 1))

        if bias:
            self.sigma_bias = nn.Parameter(th.full((output_dim,), sigma_init))

    def forward(self, observations: th.Tensor) -> th.Tensor:
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        eps_in = th.sign(self.epsilon_input.data) * th.sqrt(th.abs(self.epsilon_input.data))
        eps_out = th.sign(self.epsilon_output.data) * th.sqrt(th.abs(self.epsilon_output.data))

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()

        noise = th.mul(eps_in, eps_out)
        return nn.functional.linear(observations, self.weight + self.sigma_weight * noise, bias)
