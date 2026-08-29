import numpy as np
import torch

from rigno.models.operator import Inputs
from rigno.models.rigno import RIGNO, RegionInteractionGraphBuilder


def test_rigno_forward_and_backward():
  rng = np.random.default_rng(7)
  coordinates = rng.uniform(0, 1, size=(48, 2)).astype(np.float32)
  builder = RegionInteractionGraphBuilder(
    periodic=False,
    rmesh_levels=2,
    subsample_factor=2,
    overlap_factor_p2r=1.5,
    overlap_factor_r2p=1.5,
    node_coordinate_freqs=4,
  )
  graphs = builder.build_graphs(builder.build_metadata(
    coordinates, coordinates, np.array([[0, 0], [1, 1]], dtype=np.float32), seed=3))
  model = RIGNO(
    num_outputs=2,
    processor_steps=2,
    node_latent_size=16,
    edge_latent_size=16,
    p_edge_masking=0,
  )
  u = torch.randn(3, 1, len(coordinates), 2, requires_grad=True)
  x = torch.from_numpy(coordinates).reshape(1, 1, len(coordinates), 2).expand(3, -1, -1, -1)
  output = model(Inputs(u=u, c=None, x_inp=x, x_out=x, t=torch.zeros(3, 1),
                        tau=torch.full((3, 1), .1)), graphs)
  assert output.shape == u.shape
  output.square().mean().backward()
  assert u.grad is not None


def test_graph_builder_uses_multiscale_edges():
  rng = np.random.default_rng(2)
  coordinates = rng.uniform(0, 1, size=(64, 2)).astype(np.float32)
  domain = np.array([[0, 0], [1, 1]], dtype=np.float32)
  one = RegionInteractionGraphBuilder(False, 1, 2, 1.5, 1.5, 4)
  two = RegionInteractionGraphBuilder(False, 2, 2, 1.5, 1.5, 4)
  edges_one = one.build_metadata(coordinates, coordinates, domain, seed=5).r2r_edge_indices
  edges_two = two.build_metadata(coordinates, coordinates, domain, seed=5).r2r_edge_indices
  assert len(edges_two) >= len(edges_one)
