import numpy as np
import torch

from rigno.models.operator import Inputs
from rigno.models.rigno import RIGNO, RegionInteractionGraphBuilder


def _build_hierarchy(num_input=96, num_output=None, levels=3, seed=23):
    rng = np.random.default_rng(seed)
    x_inp = rng.uniform(0, 1, size=(num_input, 2)).astype(np.float32)
    x_out = (
        x_inp if num_output is None else rng.uniform(0, 1, size=(num_output, 2)).astype(np.float32)
    )
    builder = RegionInteractionGraphBuilder(
        periodic=False,
        rmesh_levels=levels,
        subsample_factor=2,
        overlap_factor_p2r=1.5,
        overlap_factor_r2p=1.5,
        node_coordinate_freqs=4,
    )
    metadata = builder.build_metadata(
        x_inp, x_out, np.array([[0, 0], [1, 1]], dtype=np.float32), seed=seed
    )
    return x_inp, x_out, metadata, builder.build_graphs(metadata)


def test_multigrid_builder_creates_complete_nested_hierarchy():
    _, _, metadata, graphs = _build_hierarchy()
    hierarchy = graphs.multigrid

    assert hierarchy is not None
    assert [len(points) for points in metadata.multigrid_coordinates] == [48, 24, 12]
    assert len(hierarchy.levels) == 3
    assert len(hierarchy.down_edges) == len(hierarchy.up_edges) == 2
    for level, assignment in enumerate(metadata.multigrid_assignments):
        fine_count = len(metadata.multigrid_coordinates[level])
        coarse_count = len(metadata.multigrid_coordinates[level + 1])
        assert assignment.shape == (fine_count,)
        np.testing.assert_array_equal(np.unique(assignment), np.arange(coarse_count))

        down = hierarchy.down_edges[level].edge_index
        up = hierarchy.up_edges[level].edge_index
        assert down.shape == (2, fine_count)
        torch.testing.assert_close(up, down.flip(0))
        torch.testing.assert_close(down[0], torch.arange(fine_count))
        assert down[1].min() >= 0
        assert down[1].max() < coarse_count


def test_multigrid_metadata_is_deterministic_for_a_fixed_seed():
    _, _, first, _ = _build_hierarchy(seed=29)
    _, _, second, _ = _build_hierarchy(seed=29)

    for left, right in zip(first.multigrid_coordinates, second.multigrid_coordinates):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(first.multigrid_assignments, second.multigrid_assignments):
        np.testing.assert_array_equal(left, right)


def test_multigrid_rigno_forward_backward_on_distinct_output_mesh():
    x_inp, x_out, _, graphs = _build_hierarchy(num_input=64, num_output=27, levels=2, seed=31)
    model = RIGNO(
        num_outputs=2,
        processor_steps=2,
        node_latent_size=12,
        edge_latent_size=12,
        p_edge_masking=0,
        processor_type="multigrid",
        multigrid_levels=2,
    )
    u = torch.randn(3, 1, len(x_inp), 2, requires_grad=True)
    inputs = Inputs(
        u=u,
        c=None,
        x_inp=torch.from_numpy(x_inp).reshape(1, 1, len(x_inp), 2),
        x_out=torch.from_numpy(x_out).reshape(1, 1, len(x_out), 2),
        t=torch.zeros(3, 1),
        tau=torch.full((3, 1), 0.2),
    )

    output = model(inputs, graphs)
    assert output.shape == (3, 1, len(x_out), 2)
    output.square().mean().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert model.configs["processor_type"] == "multigrid"
    assert model.intermediates["rnodes_processed"].shape[1] == len(
        graphs.multigrid.levels[0].node_features["rnodes"]
    )
