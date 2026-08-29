"""Reusable PyTorch Geometric graph-network blocks for RIGNO."""

import torch
from torch import nn
from rigno.graph.networks import InteractionNetworkLayer
from rigno.models.utils import FeedForwardBlock


class BipartiteGraphNet(nn.Module):
    """Embed and propagate information across a bipartite graph.

    Sender nodes, receiver nodes, and edges can be independently projected into
    latent spaces before one :class:`InteractionNetworkLayer` updates receiver
    nodes. Disabling an embedding is useful when that component is already in
    latent space, as for regional nodes entering the decoder.
    """

    def __init__(
        self,
        node_latent_size,
        edge_latent_size,
        mlp_hidden_layers,
        conditioned_normalization,
        cond_norm_hidden_size,
        embed_nodes=True,
        embed_sender=None,
        embed_receiver=None,
    ):
        super().__init__()
        embed_sender = embed_nodes if embed_sender is None else embed_sender
        embed_receiver = embed_nodes if embed_receiver is None else embed_receiver
        sizes = [node_latent_size] * (mlp_hidden_layers + 1)
        self.sender_embed = (
            FeedForwardBlock(sizes, use_layer_norm=True) if embed_sender else nn.Identity()
        )
        self.receiver_embed = (
            FeedForwardBlock(sizes, use_layer_norm=True) if embed_receiver else nn.Identity()
        )
        self.edge_embed = FeedForwardBlock(
            [edge_latent_size] * (mlp_hidden_layers + 1), use_layer_norm=True
        )
        self.layer = InteractionNetworkLayer(
            node_latent_size,
            edge_latent_size,
            mlp_hidden_layers,
            conditioned_normalization,
            cond_norm_hidden_size,
        )

    def forward(self, sender, receiver, edge_index, edge_attr, condition=None):
        """Return embedded senders and updated receivers.

        Node tensors have shapes ``[B, N_s, F_s]`` and ``[B, N_r, F_r]``;
        ``edge_index`` is ``[2, E]`` and ``edge_attr`` is ``[E, F_e]``.
        """
        sender, receiver = self.sender_embed(sender), self.receiver_embed(receiver)
        edge_attr = self.edge_embed(edge_attr)
        receiver, _ = self.layer(sender, receiver, edge_index, edge_attr, condition)
        return sender, receiver


class ProcessorGraphNet(nn.Module):
    """Perform repeated message passing on the regional-node graph.

    The structural edge features are embedded once. Each processor layer then
    updates both node and edge latent states, so later steps operate on the
    interactions learned by earlier steps.
    """

    def __init__(
        self,
        steps,
        node_latent_size,
        edge_latent_size,
        mlp_hidden_layers,
        conditioned_normalization,
        cond_norm_hidden_size,
    ):
        super().__init__()
        self.edge_embed = FeedForwardBlock(
            [edge_latent_size] * (mlp_hidden_layers + 1), use_layer_norm=True
        )
        self.layers = nn.ModuleList(
            [
                InteractionNetworkLayer(
                    node_latent_size,
                    edge_latent_size,
                    mlp_hidden_layers,
                    conditioned_normalization,
                    cond_norm_hidden_size,
                )
                for _ in range(steps)
            ]
        )

    def forward(self, nodes, edge_index, edge_attr, condition=None):
        """Process ``[B, N, F_n]`` nodes and return states of the same shape."""
        batched_edges = self.edge_embed(edge_attr).unsqueeze(0).expand(nodes.shape[0], -1, -1)
        for layer in self.layers:
            nodes, batched_edges = layer(nodes, nodes, edge_index, batched_edges, condition)
        return nodes
