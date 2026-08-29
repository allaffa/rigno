import torch

from rigno.models.utils import ConditionedNorm


def test_conditioned_norm_linear_biases_are_zero_initialized():
    layer = ConditionedNorm(latent_size=8, correction_size=4)

    for module in (layer.scale[0], layer.scale[2], layer.bias[0], layer.bias[2]):
        torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
