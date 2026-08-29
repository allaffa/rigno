"""PyTorch neural-network utilities."""
from collections.abc import Sequence
import torch
from torch import nn

class ConditionedNorm(nn.Module):
  def __init__(self, latent_size: int, correction_size: int):
    super().__init__()
    self.scale = nn.Sequential(nn.Linear(1, latent_size), nn.Sigmoid(), nn.Linear(latent_size, correction_size))
    self.bias = nn.Sequential(nn.Linear(1, latent_size), nn.Sigmoid(), nn.Linear(latent_size, correction_size))
    for module in (self.scale[0], self.scale[2], self.bias[0], self.bias[2]):
      nn.init.normal_(module.weight, std=0.01)
  def forward(self, x, condition):
    condition = condition.reshape(condition.shape[0], -1)[:, :1].to(dtype=x.dtype, device=x.device)
    scale, bias = 1 + condition * self.scale(condition), condition * self.bias(condition)
    while scale.ndim < x.ndim:
      scale, bias = scale.unsqueeze(1), bias.unsqueeze(1)
    return x * scale + bias

class FeedForwardBlock(nn.Module):
  def __init__(self, layer_sizes: Sequence[int], use_layer_norm=False,
               use_conditional_norm=False, cond_norm_hidden_size=16):
    super().__init__()
    if not layer_sizes:
      raise ValueError("layer_sizes cannot be empty")
    layers = []
    for index, size in enumerate(layer_sizes):
      layers.append(nn.LazyLinear(size) if index == 0 else nn.Linear(layer_sizes[index - 1], size))
      if index < len(layer_sizes) - 1:
        layers.append(nn.SiLU())
    self.layers = nn.Sequential(*layers)
    self.norm = nn.LayerNorm(layer_sizes[-1]) if use_layer_norm else nn.Identity()
    self.conditioned_norm = ConditionedNorm(cond_norm_hidden_size, layer_sizes[-1]) if use_conditional_norm else None
  def forward(self, *inputs, condition=None):
    output = self.norm(self.layers(torch.cat(inputs, dim=-1)))
    if self.conditioned_norm is not None:
      if condition is None:
        raise ValueError("condition is required when conditional normalization is enabled")
      output = self.conditioned_norm(output, condition)
    return output
