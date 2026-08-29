import torch

from rigno.graph.networks import InteractionNetworkLayer


def _fixture():
    torch.manual_seed(12)
    sender = torch.randn(2, 4, 5)
    receiver = torch.randn(2, 3, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [0, 0, 1, 1, 2]])
    edge_attr = torch.randn(5, 4)
    return sender, receiver, edge_index, edge_attr


def test_message_passing_matches_manual_mean_aggregation():
    sender, receiver, edge_index, edge_attr = _fixture()
    layer = InteractionNetworkLayer(5, 4, mlp_hidden_layers=0, conditioned_normalization=False)
    actual_nodes, actual_edges = layer(sender, receiver, edge_index, edge_attr)

    expected_nodes, expected_edges = [], []
    source, target = edge_index
    for batch_index in range(sender.shape[0]):
        edge_delta = layer.edge_mlp(
            edge_attr, sender[batch_index, source], receiver[batch_index, target]
        )
        edges = edge_attr + edge_delta
        aggregate = torch.zeros(receiver.shape[1], edges.shape[-1])
        counts = torch.zeros(receiver.shape[1], 1)
        aggregate.index_add_(0, target, edges)
        counts.index_add_(0, target, torch.ones(len(target), 1))
        aggregate = aggregate / counts.clamp_min(1)
        nodes = receiver[batch_index] + layer.node_mlp(receiver[batch_index], aggregate)
        expected_nodes.append(nodes)
        expected_edges.append(edges)

    torch.testing.assert_close(actual_nodes, torch.stack(expected_nodes))
    torch.testing.assert_close(actual_edges, torch.stack(expected_edges))


def test_message_passing_is_permutation_equivariant():
    sender, receiver, edge_index, edge_attr = _fixture()
    layer = InteractionNetworkLayer(5, 4, conditioned_normalization=False)
    expected, expected_edges = layer(sender, receiver, edge_index, edge_attr)

    sender_perm = torch.tensor([2, 0, 3, 1])
    receiver_perm = torch.tensor([2, 0, 1])
    sender_inverse = torch.argsort(sender_perm)
    receiver_inverse = torch.argsort(receiver_perm)
    permuted_edges = torch.stack([sender_inverse[edge_index[0]], receiver_inverse[edge_index[1]]])
    actual, actual_edges = layer(
        sender[:, sender_perm], receiver[:, receiver_perm], permuted_edges, edge_attr
    )

    torch.testing.assert_close(actual[:, receiver_inverse], expected)
    torch.testing.assert_close(actual_edges, expected_edges)


def test_samples_in_a_batch_do_not_interact():
    sender, receiver, edge_index, edge_attr = _fixture()
    layer = InteractionNetworkLayer(5, 4, conditioned_normalization=False)
    batched_nodes, batched_edges = layer(sender, receiver, edge_index, edge_attr)
    for index in range(sender.shape[0]):
        nodes, edges = layer(
            sender[index : index + 1], receiver[index : index + 1], edge_index, edge_attr
        )
        torch.testing.assert_close(nodes[0], batched_nodes[index])
        torch.testing.assert_close(edges[0], batched_edges[index])
