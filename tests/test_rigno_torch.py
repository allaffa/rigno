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
  assert torch.isfinite(u.grad).all()
  parameter_gradients = [parameter.grad for parameter in model.parameters()
                         if parameter.requires_grad]
  assert parameter_gradients
  assert all(gradient is not None and torch.isfinite(gradient).all()
             for gradient in parameter_gradients)

  model.eval()
  with torch.no_grad():
    first = model(Inputs(u=u.detach(), c=None, x_inp=x, x_out=x,
                         t=torch.zeros(3, 1), tau=torch.full((3, 1), .1)), graphs)
    second = model(Inputs(u=u.detach(), c=None, x_inp=x, x_out=x,
                          t=torch.zeros(3, 1), tau=torch.full((3, 1), .1)), graphs)
  torch.testing.assert_close(first, second)


def test_graph_builder_uses_multiscale_edges():
  rng = np.random.default_rng(2)
  coordinates = rng.uniform(0, 1, size=(64, 2)).astype(np.float32)
  domain = np.array([[0, 0], [1, 1]], dtype=np.float32)
  one = RegionInteractionGraphBuilder(False, 1, 2, 1.5, 1.5, 4)
  two = RegionInteractionGraphBuilder(False, 2, 2, 1.5, 1.5, 4)
  edges_one = one.build_metadata(coordinates, coordinates, domain, seed=5).r2r_edge_indices
  edges_two = two.build_metadata(coordinates, coordinates, domain, seed=5).r2r_edge_indices
  assert len(edges_two) >= len(edges_one)


def test_graph_indices_features_and_seed_are_valid():
  rng = np.random.default_rng(11)
  coordinates = rng.uniform(0, 1, size=(80, 2)).astype(np.float32)
  domain = np.array([[0, 0], [1, 1]], dtype=np.float32)
  builder = RegionInteractionGraphBuilder(True, 3, 2, 1.5, 1.5, 4)
  first = builder.build_metadata(coordinates, coordinates, domain, seed=4)
  second = builder.build_metadata(coordinates, coordinates, domain, seed=4)
  np.testing.assert_array_equal(first.x_rnodes, second.x_rnodes)
  np.testing.assert_array_equal(first.r2r_edge_indices, second.r2r_edge_indices)

  graphs = builder.build_graphs(first)
  sizes = {'pnodes': len(coordinates), 'rnodes': len(first.x_rnodes)}
  for graph in (graphs.p2r, graphs.r2r, graphs.r2p):
    for edge in graph.edges.values():
      assert edge.edge_index.dtype == torch.long
      assert edge.edge_index.shape[0] == 2
      assert edge.features.shape == (edge.edge_index.shape[1], 3)
      assert edge.edge_index[0].min() >= 0
      assert edge.edge_index[1].min() >= 0
      assert edge.edge_index[0].max() < sizes[edge.sender]
      assert edge.edge_index[1].max() < sizes[edge.receiver]
      assert torch.isfinite(edge.features).all()
  assert len(torch.unique(graphs.p2r.edge_by_name('p2r').edge_index[1])) == sizes['rnodes']
