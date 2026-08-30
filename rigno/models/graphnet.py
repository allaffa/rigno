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


class MultigridProcessorStep(nn.Module):
    """One MG-GNN step with simultaneous intra-level and cross-level updates.

    Every level processes its own graph while every adjacent pair exchanges
    learned fine-to-coarse and coarse-to-fine messages. Candidate residuals are
    fused only after all directions have been evaluated, preserving the
    paper's parallel cross-scale information flow.
    """

    def __init__(
        self,
        levels,
        node_latent_size,
        edge_latent_size,
        mlp_hidden_layers,
        conditioned_normalization,
        cond_norm_hidden_size,
    ):
        super().__init__()
        layer_args = (
            node_latent_size,
            edge_latent_size,
            mlp_hidden_layers,
            conditioned_normalization,
            cond_norm_hidden_size,
        )
        self.intra = nn.ModuleList([InteractionNetworkLayer(*layer_args) for _ in range(levels)])
        self.down = nn.ModuleList([InteractionNetworkLayer(*layer_args) for _ in range(levels - 1)])
        self.up = nn.ModuleList([InteractionNetworkLayer(*layer_args) for _ in range(levels - 1)])
        sizes = [node_latent_size] * (mlp_hidden_layers + 1)
        self.fusion = nn.ModuleList(
            [FeedForwardBlock(sizes, use_layer_norm=True) for _ in range(levels)]
        )

    def forward(
        self,
        nodes,
        intra_indices,
        intra_edges,
        down_indices,
        down_edges,
        up_indices,
        up_edges,
        condition=None,
    ):
        """Update all hierarchy levels and latent edge states once."""
        candidates = [[] for _ in nodes]
        new_intra, new_down, new_up = [], [], []
        for level, layer in enumerate(self.intra):
            updated, edge_state = layer(
                nodes[level],
                nodes[level],
                intra_indices[level],
                intra_edges[level],
                condition,
            )
            candidates[level].append(updated - nodes[level])
            new_intra.append(edge_state)
        for level, layer in enumerate(self.down):
            updated, edge_state = layer(
                nodes[level],
                nodes[level + 1],
                down_indices[level],
                down_edges[level],
                condition,
            )
            candidates[level + 1].append(updated - nodes[level + 1])
            new_down.append(edge_state)
        for level, layer in enumerate(self.up):
            updated, edge_state = layer(
                nodes[level + 1],
                nodes[level],
                up_indices[level],
                up_edges[level],
                condition,
            )
            candidates[level].append(updated - nodes[level])
            new_up.append(edge_state)
        updated_nodes = [
            node + fusion(*level_candidates)
            for node, fusion, level_candidates in zip(nodes, self.fusion, candidates)
        ]
        return updated_nodes, new_intra, new_down, new_up


class MultigridProcessorGraphNet(nn.Module):
    """MG-GNN regional processor with explicit parallel graph levels.

    Coarse latent states are initialized by mean restriction along assignment
    edges. Each processing step then performs homogeneous message passing on
    every level and heterogeneous message passing in both directions between
    adjacent levels. The final finest-level state is returned to RIGNO's decoder.
    """

    def __init__(
        self,
        steps,
        levels,
        node_latent_size,
        edge_latent_size,
        mlp_hidden_layers,
        conditioned_normalization,
        cond_norm_hidden_size,
    ):
        super().__init__()
        if levels < 2:
            raise ValueError("multigrid processor requires at least two levels")
        self.levels = levels
        edge_sizes = [edge_latent_size] * (mlp_hidden_layers + 1)
        self.intra_edge_embed = nn.ModuleList(
            [FeedForwardBlock(edge_sizes, use_layer_norm=True) for _ in range(levels)]
        )
        self.down_edge_embed = nn.ModuleList(
            [FeedForwardBlock(edge_sizes, use_layer_norm=True) for _ in range(levels - 1)]
        )
        self.up_edge_embed = nn.ModuleList(
            [FeedForwardBlock(edge_sizes, use_layer_norm=True) for _ in range(levels - 1)]
        )
        self.steps = nn.ModuleList(
            [
                MultigridProcessorStep(
                    levels,
                    node_latent_size,
                    edge_latent_size,
                    mlp_hidden_layers,
                    conditioned_normalization,
                    cond_norm_hidden_size,
                )
                for _ in range(steps)
            ]
        )
        readout_args = (
            node_latent_size,
            edge_latent_size,
            mlp_hidden_layers,
            conditioned_normalization,
            cond_norm_hidden_size,
        )
        self.readout_up = nn.ModuleList(
            [InteractionNetworkLayer(*readout_args) for _ in range(levels - 1)]
        )

    @staticmethod
    def _restrict(fine_nodes, edge_index, coarse_count):
        """Mean-pool fine states to coarse clusters defined by assignment edges."""
        target = edge_index[1]
        coarse = fine_nodes.new_zeros(fine_nodes.shape[0], coarse_count, fine_nodes.shape[-1])
        counts = fine_nodes.new_zeros(coarse_count)
        coarse.index_add_(1, target, fine_nodes[:, edge_index[0]])
        counts.index_add_(0, target, torch.ones_like(target, dtype=fine_nodes.dtype))
        return coarse / counts.clamp_min(1).view(1, -1, 1)

    def forward(self, fine_nodes, hierarchy, condition=None, edge_masker=None):
        """Process a regional hierarchy and return its finest latent states.

        Args:
            fine_nodes: Encoded finest regional states ``[B, N_0, F_n]``.
            hierarchy: :class:`MultigridGraphSet` with at least ``levels`` levels.
            condition: Optional per-sample scalar conditioning tensor.
            edge_masker: Optional callable returning a masked ``(index, features)``
                pair for training-time edge regularization.
        """
        if len(hierarchy.levels) < self.levels:
            raise ValueError(
                f"multigrid processor expects {self.levels} levels, "
                f"but the graph hierarchy has {len(hierarchy.levels)}"
            )

        def edge_data(edge):
            return (edge.edge_index, edge.features) if edge_masker is None else edge_masker(edge)

        intra_data = [
            edge_data(graph.edge_by_name("r2r")) for graph in hierarchy.levels[: self.levels]
        ]
        # Assignment edges define restriction/prolongation coverage and therefore
        # are never randomly masked.
        down_data = [
            (edge.edge_index, edge.features) for edge in hierarchy.down_edges[: self.levels - 1]
        ]
        up_data = [
            (edge.edge_index, edge.features) for edge in hierarchy.up_edges[: self.levels - 1]
        ]
        intra_indices = [item[0] for item in intra_data]
        down_indices = [item[0] for item in down_data]
        up_indices = [item[0] for item in up_data]
        nodes = [fine_nodes]
        for level, edge_index in enumerate(down_indices):
            coarse_count = hierarchy.levels[level + 1].node_features["rnodes"].shape[0]
            nodes.append(self._restrict(nodes[-1], edge_index, coarse_count))
        intra_edges = [
            embed(item[1].to(fine_nodes.dtype)).unsqueeze(0).expand(fine_nodes.shape[0], -1, -1)
            for embed, item in zip(self.intra_edge_embed, intra_data)
        ]
        down_edges = [
            embed(item[1].to(fine_nodes.dtype)).unsqueeze(0).expand(fine_nodes.shape[0], -1, -1)
            for embed, item in zip(self.down_edge_embed, down_data)
        ]
        up_edges = [
            embed(item[1].to(fine_nodes.dtype)).unsqueeze(0).expand(fine_nodes.shape[0], -1, -1)
            for embed, item in zip(self.up_edge_embed, up_data)
        ]
        for step in self.steps:
            nodes, intra_edges, down_edges, up_edges = step(
                nodes,
                intra_indices,
                intra_edges,
                down_indices,
                down_edges,
                up_indices,
                up_edges,
                condition,
            )
        # Consume the final state of every coarse level. Without this readout,
        # coarse updates from the last parallel MG-GNN step would not influence
        # a loss defined only on RIGNO's finest regional/output nodes.
        for level in reversed(range(self.levels - 1)):
            nodes[level], _ = self.readout_up[level](
                nodes[level + 1],
                nodes[level],
                up_indices[level],
                up_edges[level],
                condition,
            )
        return nodes[0]
