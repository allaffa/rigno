"""Reusable PyTorch Geometric graph-network blocks for RIGNO."""
import torch
from torch import nn
from rigno.graph.networks import InteractionNetworkLayer
from rigno.models.utils import FeedForwardBlock

class BipartiteGraphNet(nn.Module):
  def __init__(self, node_latent_size, edge_latent_size, mlp_hidden_layers,
               conditioned_normalization, cond_norm_hidden_size, embed_nodes=True):
    super().__init__()
    sizes = [node_latent_size] * (mlp_hidden_layers + 1)
    self.sender_embed = FeedForwardBlock(sizes, use_layer_norm=True) if embed_nodes else nn.Identity()
    self.receiver_embed = FeedForwardBlock(sizes, use_layer_norm=True) if embed_nodes else nn.Identity()
    self.edge_embed = FeedForwardBlock([edge_latent_size] * (mlp_hidden_layers + 1), use_layer_norm=True)
    self.layer = InteractionNetworkLayer(node_latent_size, edge_latent_size, mlp_hidden_layers,
      conditioned_normalization, cond_norm_hidden_size)

  def forward(self, sender, receiver, edge_index, edge_attr, condition=None):
    sender, receiver = self.sender_embed(sender), self.receiver_embed(receiver)
    edge_attr = self.edge_embed(edge_attr)
    receiver, _ = self.layer(sender, receiver, edge_index, edge_attr, condition)
    return sender, receiver

class ProcessorGraphNet(nn.Module):
  def __init__(self, steps, node_latent_size, edge_latent_size, mlp_hidden_layers,
               conditioned_normalization, cond_norm_hidden_size):
    super().__init__()
    self.edge_embed = FeedForwardBlock([edge_latent_size] * (mlp_hidden_layers + 1), use_layer_norm=True)
    self.layers = nn.ModuleList([InteractionNetworkLayer(
      node_latent_size, edge_latent_size, mlp_hidden_layers,
      conditioned_normalization, cond_norm_hidden_size) for _ in range(steps)])

  def forward(self, nodes, edge_index, edge_attr, condition=None):
    batched_edges = self.edge_embed(edge_attr).unsqueeze(0).expand(nodes.shape[0], -1, -1)
    for layer in self.layers:
      node_parts, edge_parts = [], []
      for batch_index in range(nodes.shape[0]):
        cond = None if condition is None else condition[batch_index:batch_index + 1]
        node, edge = layer(nodes[batch_index:batch_index + 1],
                           nodes[batch_index:batch_index + 1], edge_index,
                           batched_edges[batch_index], cond)
        node_parts.append(node[0])
        edge_parts.append(edge[0])
      nodes, batched_edges = torch.stack(node_parts), torch.stack(edge_parts)
    return nodes
