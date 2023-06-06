import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import OrderedDict

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        nhidden: tuple = (64, 64),
        nonlinearity: str = "relu",
    ):
        super().__init__()
        assert len(nhidden) >= 1
        if nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "leaky_relu":
            self.nonlinearity = nn.LeakyReLU()
        elif nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()
        else:
            raise ValueError(f"nonlinearity {nonlinearity} not supported")

        di = {
            "batchnorm": nn.BatchNorm1d(nin),
            "hidden_layer_0": nn.Linear(nin, nhidden[0], bias=True),
            "nonlinearity_0": self.nonlinearity,
        }
        nn.init.xavier_uniform_(
            di["hidden_layer_0"].weight, gain=nn.init.calculate_gain(nonlinearity)
        )
        for i in range(1, len(nhidden)):
            di.update(
                {
                    f"hidden_layer_{i}": nn.Linear(
                        nhidden[i - 1], nhidden[i], bias=True
                    ),
                    f"nonlinearity_{i}": self.nonlinearity,
                }
            )
            nn.init.xavier_uniform_(
                di[f"hidden_layer_{i}"].weight,
                gain=nn.init.calculate_gain(nonlinearity),
            )
        di.update(
            {
                "output_layer": nn.Linear(nhidden[-1], nout, bias=True),
            }
        )
        nn.init.xavier_uniform_(
            di["output_layer"].weight, gain=nn.init.calculate_gain(nonlinearity)
        )

        self.net = nn.Sequential(OrderedDict(di))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)
