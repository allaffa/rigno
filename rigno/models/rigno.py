"""PyTorch/PyG implementation of the Region Interaction Graph Neural Operator."""

from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import Delaunay
import torch

from rigno.graph.entities import EdgeSet, TypedGraph
from rigno.models.graphnet import BipartiteGraphNet, ProcessorGraphNet
from rigno.models.operator import AbstractOperator, Inputs
from rigno.models.utils import FeedForwardBlock


@dataclass(frozen=True)
class RegionInteractionGraphMetadata:
    x_pnodes_inp: np.ndarray
    x_pnodes_out: np.ndarray
    x_rnodes: np.ndarray
    r_rnodes: np.ndarray
    p2r_edge_indices: np.ndarray
    r2r_edge_indices: np.ndarray
    r2r_edge_domains: np.ndarray
    r2p_edge_indices: np.ndarray


@dataclass(frozen=True)
class RegionInteractionGraphSet:
    p2r: TypedGraph
    r2r: TypedGraph
    r2p: TypedGraph

    def to(self, device):
        return RegionInteractionGraphSet(
            self.p2r.to(device), self.r2r.to(device), self.r2p.to(device)
        )


class RegionInteractionGraphBuilder:
    """Construct physical/regional multiscale graphs using NumPy and SciPy."""

    _domain_shifts = np.array(
        [[0, 0], [-2, 0], [-2, 2], [0, 2], [2, 2], [2, 0], [2, -2], [0, -2], [-2, -2]],
        dtype=np.float32,
    )

    def __init__(
        self,
        periodic,
        rmesh_levels,
        subsample_factor,
        overlap_factor_p2r,
        overlap_factor_r2p,
        node_coordinate_freqs,
    ):
        self.periodic = periodic
        self.rmesh_levels = rmesh_levels
        self.subsample_factor = subsample_factor
        self.overlap_factor_p2r = overlap_factor_p2r
        self.overlap_factor_r2p = overlap_factor_r2p
        self.node_coordinate_freqs = node_coordinate_freqs

    def _support_radii(self, x):
        extended = (x[None] + self._domain_shifts[:, None]).reshape(-1, 2) if self.periodic else x
        tri = Delaunay(extended)
        medians = _compute_triangulation_medians(tri)
        radii = np.zeros(len(x), dtype=np.float32)
        mask = tri.simplices < len(x)
        values, indices = medians[mask], tri.simplices[mask]
        order = np.argsort(indices)
        unique, starts = np.unique(indices[order], return_index=True)
        radii[unique] = np.maximum.reduceat(values[order], starts)
        return radii

    def _support_edges(self, centers, points, radii):
        rel = points[:, None] - centers[None]
        if self.periodic:
            rel = np.where(rel >= 1, rel - 2, rel)
            rel = np.where(rel < -1, rel + 2, rel)
        radii = np.where(radii < 0.5, radii, 0.2)
        return np.stack(np.where(np.linalg.norm(rel, axis=-1) <= radii[None]), axis=-1)

    def _r2r_edges(self, x):
        edge_parts, domain_parts = [], []
        for level in range(self.rmesh_levels):
            size = int(len(x) / self.subsample_factor**level)
            if size < 4:
                continue
            points = x[:size]
            extended = (
                (points[None] + self._domain_shifts[:, None]).reshape(-1, 2)
                if self.periodic
                else points
            )
            edges = _edges_from_triangulation(Delaunay(extended))
            domains, local = edges // size, edges % size
            keep = np.any(domains == 0, axis=1) if self.periodic else np.all(domains == 0, axis=1)
            edge_parts.append(local[keep])
            domain_parts.append(domains[keep])
        if not edge_parts:
            raise ValueError("regional mesh is too small to triangulate")
        edges, domains = np.concatenate(edge_parts), np.concatenate(domain_parts)
        _, unique = np.unique(edges, axis=0, return_index=True)
        return edges[unique], domains[unique]

    def build_metadata(self, x_inp, x_out, domain, rmesh_correction_dsf=1, seed=0):
        x_inp, x_out, domain = map(
            lambda v: np.asarray(v, dtype=np.float32), (x_inp, x_out, domain)
        )
        x_inp = 2 * (x_inp - domain[0]) / (domain[1] - domain[0]) - 1
        x_out = 2 * (x_out - domain[0]) / (domain[1] - domain[0]) - 1
        rng = np.random.default_rng(seed)
        x_rnodes = x_inp[rng.permutation(len(x_inp))[: int(len(x_inp) / self.subsample_factor)]]
        if rmesh_correction_dsf > 1:
            x_rnodes = x_rnodes[
                rng.permutation(len(x_rnodes))[: int(len(x_rnodes) / rmesh_correction_dsf)]
            ]
        elif rmesh_correction_dsf < 1:
            x_rnodes = _upsample(x_rnodes, 1 / rmesh_correction_dsf, rng)
        radii = self._support_radii(x_rnodes)
        p2r = self._support_edges(
            x_rnodes, x_inp, np.clip(self.overlap_factor_p2r * radii, 0, radii.max())
        )
        r2r, domains = self._r2r_edges(x_rnodes)
        r2p = np.flip(
            self._support_edges(
                x_rnodes, x_out, np.clip(self.overlap_factor_r2p * radii, 0, radii.max())
            ),
            axis=-1,
        )
        return RegionInteractionGraphMetadata(x_inp, x_out, x_rnodes, radii, p2r, r2r, domains, r2p)

    def _node_features(self, x, radius=None):
        if self.periodic:
            phi = math.pi * (x + 1)
            freqs = np.arange(1, self.node_coordinate_freqs + 1)
            features = np.concatenate(
                [
                    np.sin(phi[..., None] * freqs).reshape(len(x), -1),
                    np.cos(phi[..., None] * freqs).reshape(len(x), -1),
                ],
                axis=-1,
            )
        else:
            features = x
        if radius is not None:
            features = np.concatenate([features, radius[:, None]], axis=-1)
        return torch.as_tensor(features, dtype=torch.float32)

    def _edge_set(self, sender_name, receiver_name, x_sender, x_receiver, indices, domains=None):
        source, target = indices[:, 0], indices[:, 1]
        rel = x_sender[source] - x_receiver[target]
        if self.periodic:
            if domains is None:
                rel = np.where(rel < -1, rel + 2, rel)
                rel = np.where(rel >= 1, rel - 2, rel)
            else:
                rel = (
                    x_sender[source]
                    + self._domain_shifts[domains[:, 0]]
                    - x_receiver[target]
                    - self._domain_shifts[domains[:, 1]]
                )
        length = 2 * math.sqrt(x_sender.shape[-1])
        features = np.concatenate(
            [rel / length, np.linalg.norm(rel, axis=-1, keepdims=True) / length], axis=-1
        )
        return EdgeSet(
            torch.as_tensor(indices.T.copy(), dtype=torch.long),
            torch.as_tensor(features, dtype=torch.float32),
            sender_name,
            receiver_name,
        )

    def build_graphs(self, metadata):
        xp, xo, xr, rr = (
            metadata.x_pnodes_inp,
            metadata.x_pnodes_out,
            metadata.x_rnodes,
            metadata.r_rnodes,
        )
        pnodes_in, pnodes_out, rnodes = (
            self._node_features(xp),
            self._node_features(xo),
            self._node_features(xr, rr),
        )
        return RegionInteractionGraphSet(
            TypedGraph(
                {"pnodes": pnodes_in, "rnodes": rnodes},
                {"p2r": self._edge_set("pnodes", "rnodes", xp, xr, metadata.p2r_edge_indices)},
            ),
            TypedGraph(
                {"rnodes": rnodes},
                {
                    "r2r": self._edge_set(
                        "rnodes",
                        "rnodes",
                        xr,
                        xr,
                        metadata.r2r_edge_indices,
                        metadata.r2r_edge_domains,
                    )
                },
            ),
            TypedGraph(
                {"rnodes": rnodes, "pnodes": pnodes_out},
                {"r2p": self._edge_set("rnodes", "pnodes", xr, xo, metadata.r2p_edge_indices)},
            ),
        )


class RIGNO(AbstractOperator):
    """Encode-process-decode RIGNO implemented as a torch module."""

    def __init__(
        self,
        num_outputs,
        processor_steps=18,
        node_latent_size=128,
        edge_latent_size=128,
        mlp_hidden_layers=1,
        concatenate_t=True,
        concatenate_tau=True,
        conditioned_normalization=True,
        cond_norm_hidden_size=16,
        p_edge_masking=0.5,
    ):
        super().__init__()
        self.num_outputs, self.concatenate_t, self.concatenate_tau = (
            num_outputs,
            concatenate_t,
            concatenate_tau,
        )
        self.conditioned_normalization, self.p_edge_masking = (
            conditioned_normalization,
            p_edge_masking,
        )
        common = (
            node_latent_size,
            edge_latent_size,
            mlp_hidden_layers,
            conditioned_normalization,
            cond_norm_hidden_size,
        )
        self.encoder = BipartiteGraphNet(*common)
        self.processor = ProcessorGraphNet(processor_steps, *common)
        self.decoder = BipartiteGraphNet(
            *common, embed_nodes=False, embed_sender=False, embed_receiver=True
        )
        self.output_projection = FeedForwardBlock(
            [node_latent_size] * mlp_hidden_layers + [num_outputs]
        )
        self._configs = dict(
            num_outputs=num_outputs,
            processor_steps=processor_steps,
            node_latent_size=node_latent_size,
            edge_latent_size=edge_latent_size,
            mlp_hidden_layers=mlp_hidden_layers,
            concatenate_t=concatenate_t,
            concatenate_tau=concatenate_tau,
            conditioned_normalization=conditioned_normalization,
            cond_norm_hidden_size=cond_norm_hidden_size,
            p_edge_masking=p_edge_masking,
        )
        self.intermediates = {}

    @staticmethod
    def _channel(value, batch, device, dtype):
        if value is None:
            return None
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.numel() == 1:
            value = value.expand(batch)
        return value.reshape(batch, -1)[:, :1]

    def _masked(self, edge):
        if not self.training or self.p_edge_masking <= 0:
            return edge.edge_index, edge.features
        keep = max(1, int((1 - self.p_edge_masking) * edge.features.shape[0]))
        chosen = torch.randperm(edge.features.shape[0], device=edge.features.device)[:keep]
        return edge.edge_index[:, chosen], edge.features[chosen]

    def forward(self, inputs: Inputs, graphs: RegionInteractionGraphSet):
        if inputs.u.ndim != 4 or inputs.u.shape[1] != 1:
            raise ValueError("inputs.u must have shape [batch, 1, points, channels]")
        batch, _, points, _ = inputs.u.shape
        device, dtype = inputs.u.device, inputs.u.dtype
        graph_device = graphs.p2r.node_features["pnodes"].device
        if graph_device != device:
            graphs = graphs.to(device)
        for name, coordinates, expected in (
            ("x_inp", inputs.x_inp, graphs.p2r.node_features["pnodes"].shape[0]),
            ("x_out", inputs.x_out, graphs.r2p.node_features["pnodes"].shape[0]),
        ):
            if coordinates is None or coordinates.ndim != 4:
                raise ValueError(f"inputs.{name} must have shape [batch, 1, points, dimensions]")
            if coordinates.shape[0] not in (1, batch) or coordinates.shape[1] != 1:
                raise ValueError(f"inputs.{name} must have shape [batch, 1, points, dimensions]")
            if coordinates.shape[2] != expected:
                raise ValueError(
                    f"inputs.{name} has {coordinates.shape[2]} points, "
                    f"but its graph expects {expected}"
                )
        features = inputs.u[:, 0]
        input_node_count = graphs.p2r.node_features["pnodes"].shape[0]
        if points != input_node_count:
            raise ValueError(
                f"inputs.u has {points} points, but the p2r graph expects {input_node_count}"
            )
        if inputs.c is not None:
            if inputs.c.ndim != 4 or inputs.c.shape[:3] != inputs.u.shape[:3]:
                raise ValueError(
                    "inputs.c must have shape [batch, 1, points, channels] matching inputs.u"
                )
            features = torch.cat([features, inputs.c[:, 0]], dim=-1)
        t, tau = self._channel(inputs.t, batch, device, dtype), self._channel(
            inputs.tau, batch, device, dtype
        )
        forced = []
        if self.concatenate_t:
            if t is None:
                raise ValueError("t is required when concatenate_t=True")
            forced.append(t[:, None].expand(-1, points, -1))
        if self.concatenate_tau:
            if tau is None:
                raise ValueError("tau is required when concatenate_tau=True")
            forced.append(tau[:, None].expand(-1, points, -1))
        if forced:
            features = torch.cat([features, *forced], dim=-1)
        condition = tau if self.conditioned_normalization else None
        if condition is None and self.conditioned_normalization:
            condition = torch.zeros((batch, 1), device=device, dtype=dtype)
        p_struct = graphs.p2r.node_features["pnodes"].to(dtype).unsqueeze(0).expand(batch, -1, -1)
        r_struct = graphs.p2r.node_features["rnodes"].to(dtype).unsqueeze(0).expand(batch, -1, -1)
        edge = graphs.p2r.edge_by_name("p2r")
        edge_index, edge_attr = self._masked(edge)
        latent_p, latent_r = self.encoder(
            torch.cat([features, p_struct], -1),
            r_struct,
            edge_index,
            edge_attr.to(dtype),
            condition,
        )
        encoded_r = latent_r
        edge = graphs.r2r.edge_by_name("r2r")
        edge_index, edge_attr = self._masked(edge)
        latent_r = self.processor(latent_r, edge_index, edge_attr.to(dtype), condition)
        edge = graphs.r2p.edge_by_name("r2p")
        edge_index, edge_attr = self._masked(edge)
        output_struct = (
            graphs.r2p.node_features["pnodes"].to(dtype).unsqueeze(0).expand(batch, -1, -1)
        )
        _, decoded = self.decoder(
            latent_r, output_struct, edge_index, edge_attr.to(dtype), condition
        )
        self.intermediates = {
            "pnodes_encoded": latent_p,
            "rnodes_encoded": encoded_r,
            "rnodes_processed": latent_r,
            "pnodes_decoded": decoded,
        }
        return self.output_projection(decoded).unsqueeze(1)


def _edges_from_triangulation(tri):
    indptr, cols = tri.vertex_neighbor_vertices
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    edges = np.stack([rows, cols], axis=-1)
    return np.concatenate([edges, np.flip(edges, axis=-1)], axis=0)


def _compute_triangulation_medians(tri):
    edges, medians = np.zeros(tri.simplices.shape), np.zeros(tri.simplices.shape)
    for i in range(tri.simplices.shape[1]):
        left, right = [
            part.squeeze(1)
            for part in np.split(tri.points[np.delete(tri.simplices, i, axis=1)], 2, axis=1)
        ]
        edges[:, i] = np.linalg.norm(left - right, axis=1)
    for i in range(tri.simplices.shape[1]):
        value = (2 * np.sum(np.delete(edges, i, axis=1) ** 2, axis=1) - edges[:, i] ** 2) / 4
        medians[:, i] = 0.67 * np.sqrt(np.maximum(value, 0))
    return medians


def _upsample(x, factor, rng):
    count = int(len(x) * (factor ** x.shape[-1] - 1))
    simplices = rng.permutation(Delaunay(x).simplices)[:count]
    return np.concatenate([x, np.mean(x[simplices], axis=1)])
