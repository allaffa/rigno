"""Abstract learnable operator interfaces."""

from typing import NamedTuple
import torch
from torch import nn


class Inputs(NamedTuple):
    """Inputs consumed by a neural PDE solution operator.

    ``u`` and optional coefficients ``c`` use shape ``[B, 1, N_in, C]``.
    ``x_inp`` and ``x_out`` contain input/output coordinates with shape
    ``[B or 1, 1, N, D]``. ``t`` and ``tau`` are optional current-time and
    lead-time values, normally one scalar per batch sample.
    """

    u: torch.Tensor
    c: torch.Tensor | None
    x_inp: torch.Tensor
    x_out: torch.Tensor
    t: torch.Tensor | float | None
    tau: torch.Tensor | float | None


class AbstractOperator(nn.Module):
    """Base class for PyTorch neural operators with serializable configuration."""

    @property
    def configs(self):
        """Return a detached dictionary of constructor configuration values."""
        return dict(self._configs)
