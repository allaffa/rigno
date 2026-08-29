"""Lightweight graph containers used by the PyTorch Geometric RIGNO port."""

from dataclasses import dataclass
from typing import Dict
import torch


@dataclass(frozen=True)
class EdgeSet:
    """One directed edge type in a typed graph.

    Attributes:
        edge_index: Integer tensor of shape ``[2, E]``. Row zero contains
            sender-node indices and row one contains receiver-node indices.
        features: Structural or latent edge features of shape ``[E, F_e]``.
        sender: Name of the sender node set in the parent graph.
        receiver: Name of the receiver node set in the parent graph.
    """

    edge_index: torch.Tensor
    features: torch.Tensor
    sender: str
    receiver: str

    def to(self, device):
        """Return a copy whose index and feature tensors are on ``device``."""
        return EdgeSet(
            self.edge_index.to(device), self.features.to(device), self.sender, self.receiver
        )


@dataclass(frozen=True)
class TypedGraph:
    """Graph container with named node sets and named directed edge sets.

    ``node_features`` maps each node-set name to a tensor of shape ``[N, F_n]``.
    Every :class:`EdgeSet` refers to two of those names through its ``sender``
    and ``receiver`` fields.
    """

    node_features: Dict[str, torch.Tensor]
    edges: Dict[str, EdgeSet]

    def edge_by_name(self, name: str) -> EdgeSet:
        """Return the edge set named ``name`` or raise ``KeyError``."""
        return self.edges[name]

    def to(self, device):
        """Return a graph with every contained tensor moved to ``device``."""
        return TypedGraph(
            {key: value.to(device) for key, value in self.node_features.items()},
            {key: value.to(device) for key, value in self.edges.items()},
        )
