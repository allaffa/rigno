"""Abstract learnable operator interfaces."""
from typing import NamedTuple
import torch
from torch import nn

class Inputs(NamedTuple):
  u: torch.Tensor
  c: torch.Tensor | None
  x_inp: torch.Tensor
  x_out: torch.Tensor
  t: torch.Tensor | float | None
  tau: torch.Tensor | float | None

class AbstractOperator(nn.Module):
  @property
  def configs(self):
    return dict(self._configs)
