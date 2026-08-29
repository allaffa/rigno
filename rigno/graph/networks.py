"""PyTorch Geometric interaction-network layers."""

import torch
from torch_geometric.nn import MessagePassing
from rigno.models.utils import FeedForwardBlock


class InteractionNetworkLayer(MessagePassing):
    """Edge-first interaction network with mean aggregation and residual updates."""

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
        return edge_attr
