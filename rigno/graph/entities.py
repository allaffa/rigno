"""Lightweight graph containers used by the PyTorch Geometric RIGNO port."""

from dataclasses import dataclass
from typing import Dict
import torch


@dataclass(frozen=True)
class EdgeSet:
    edge_index: torch.Tensor
    features: torch.Tensor
    sender: str
    receiver: str

    def to(self, device):
        return EdgeSet(
            self.edge_index.to(device), self.features.to(device), self.sender, self.receiver
        )


@dataclass(frozen=True)
class TypedGraph:
    node_features: Dict[str, torch.Tensor]
    edges: Dict[str, EdgeSet]

    def edge_by_name(self, name: str) -> EdgeSet:
        return self.edges[name]

    def to(self, device):
        return TypedGraph(
            {key: value.to(device) for key, value in self.node_features.items()},
            {key: value.to(device) for key, value in self.edges.items()},
        )
