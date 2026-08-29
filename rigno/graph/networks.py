"""PyTorch Geometric interaction-network layers."""

import torch
from torch_geometric.nn import MessagePassing
from rigno.models.utils import FeedForwardBlock


class InteractionNetworkLayer(MessagePassing):
    """Apply one residual interaction-network message-passing step.

    The layer first updates every edge from its current features and the latent
    states of its sender and receiver. Updated edge features are then averaged
    at each receiver node, and a node MLP produces a residual receiver update.
    A shared graph topology is vectorized across the batch using offset indices.

    Args:
        node_latent_size: Width of sender and receiver latent states.
        edge_latent_size: Width of latent edge states.
        mlp_hidden_layers: Number of hidden transformations in each update MLP.
        conditioned_normalization: Whether update MLP outputs are conditioned
            on a scalar such as the requested time increment.
        cond_norm_hidden_size: Hidden width of the conditioning networks.
    """

    def __init__(
        self,
        node_latent_size,
        edge_latent_size,
        mlp_hidden_layers=1,
        conditioned_normalization=True,
        cond_norm_hidden_size=16,
    ):
        super().__init__(aggr="mean", flow="source_to_target")
        kwargs = dict(
            use_layer_norm=True,
            use_conditional_norm=conditioned_normalization,
            cond_norm_hidden_size=cond_norm_hidden_size,
        )
        self.edge_mlp = FeedForwardBlock([edge_latent_size] * (mlp_hidden_layers + 1), **kwargs)
        self.node_mlp = FeedForwardBlock([node_latent_size] * (mlp_hidden_layers + 1), **kwargs)

    def forward(self, sender, receiver, edge_index, edge_attr, condition=None):
        """Update edges and receiver nodes.

        Args:
            sender: Sender-node states with shape ``[B, N_s, F_n]``.
            receiver: Receiver-node states with shape ``[B, N_r, F_n]``.
            edge_index: Shared directed topology with shape ``[2, E]``.
            edge_attr: Edge states with shape ``[E, F_e]`` (shared across the
                batch) or ``[B, E, F_e]`` (one state per sample).
            condition: Optional per-sample conditioning values of shape
                ``[B, 1]`` or any shape flattenable to one scalar per sample.

        Returns:
            A pair ``(receiver_states, edge_states)`` with shapes
            ``[B, N_r, F_n]`` and ``[B, E, F_e]`` respectively.
        """
        batch_size, num_senders, _ = sender.shape
        num_receivers = receiver.shape[1]
        num_edges = edge_index.shape[1]
        if edge_attr.ndim == 2:
            edge_attr = edge_attr.unsqueeze(0).expand(batch_size, -1, -1)
        elif edge_attr.shape[0] != batch_size:
            raise ValueError("batched edge_attr must have the same batch size as the nodes")

        offsets = torch.arange(batch_size, device=edge_index.device)[:, None]
        source = edge_index[0].unsqueeze(0) + offsets * num_senders
        target = edge_index[1].unsqueeze(0) + offsets * num_receivers
        batched_edge_index = torch.stack([source.reshape(-1), target.reshape(-1)])
        sender_flat = sender.reshape(batch_size * num_senders, -1)
        receiver_flat = receiver.reshape(batch_size * num_receivers, -1)
        edges_flat = edge_attr.reshape(batch_size * num_edges, -1)
        edge_condition = (
            None if condition is None else condition.repeat_interleave(num_edges, dim=0)
        )
        delta = self.edge_mlp(
            edges_flat,
            sender_flat[batched_edge_index[0]],
            receiver_flat[batched_edge_index[1]],
            condition=edge_condition,
        )
        updated_edges = edges_flat + delta
        aggregated = self.propagate(
            edge_index=batched_edge_index,
            edge_attr=updated_edges,
            size=(batch_size * num_senders, batch_size * num_receivers),
        )
        node_condition = (
            None if condition is None else condition.repeat_interleave(num_receivers, dim=0)
        )
        updated_receivers = receiver_flat + self.node_mlp(
            receiver_flat, aggregated, condition=node_condition
        )
        return updated_receivers.reshape(batch_size, num_receivers, -1), updated_edges.reshape(
            batch_size, num_edges, -1
        )

    def message(self, edge_attr):
        """Use updated edge states directly as messages for mean aggregation."""
        return edge_attr
