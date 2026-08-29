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
        outputs, edges = [], []
        for batch_index in range(sender.shape[0]):
            cond = None if condition is None else condition[batch_index : batch_index + 1]
            source, target = edge_index
            delta = self.edge_mlp(
                edge_attr,
                sender[batch_index, source],
                receiver[batch_index, target],
                condition=cond,
            )
            updated_edges = edge_attr + delta
            aggregated = self.propagate(
                edge_index=edge_index,
                edge_attr=updated_edges,
                size=(sender.shape[1], receiver.shape[1]),
            )
            node_delta = self.node_mlp(receiver[batch_index], aggregated, condition=cond)
            outputs.append(receiver[batch_index] + node_delta)
            edges.append(updated_edges)
        return torch.stack(outputs), torch.stack(edges)

    def message(self, edge_attr):
        return edge_attr
