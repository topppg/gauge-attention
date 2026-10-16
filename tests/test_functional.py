import pytest
import torch

from gauge_attention import (
    connection_to_absolute_phases,
    pairwise_transport_phases,
    rope_connection,
    rope_frequencies,
    rotate_pairs,
)


def test_connection_accumulates_from_zero() -> None:
    connection = torch.tensor([[[0.2, -0.1, 0.4]]])
    phases = connection_to_absolute_phases(connection)
    expected = torch.tensor([[[0.0], [0.2], [0.1], [0.5]]])
    torch.testing.assert_close(phases, expected)


def test_transport_is_antisymmetric_and_composes() -> None:
    torch.manual_seed(1)
    connection = torch.randn(2, 3, 5)
    transport = pairwise_transport_phases(connection_to_absolute_phases(connection))
    torch.testing.assert_close(transport, -transport.transpose(1, 2))
    torch.testing.assert_close(transport[:, 1, 4] + transport[:, 4, 5], transport[:, 1, 5])


def test_nonconstant_connection_is_not_distance_only() -> None:
    connection = torch.tensor([[[0.1, 0.4, -0.2]]])
    transport = pairwise_transport_phases(connection_to_absolute_phases(connection))
    assert transport[0, 0, 1, 0] != transport[0, 1, 2, 0]


def test_constant_connection_is_rope() -> None:
    connection = rope_connection(2, 8, 7, base=1_000.0)
    phases = connection_to_absolute_phases(connection)
    frequencies = rope_frequencies(8, base=1_000.0)
    expected = torch.arange(7)[:, None] * frequencies[None, :]
    torch.testing.assert_close(phases, expected[None, ...].expand(2, -1, -1))


def test_rotation_preserves_norm() -> None:
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    phases = torch.randn(2, 3, 4, dtype=torch.float64)
    torch.testing.assert_close(rotate_pairs(x, phases).norm(dim=-1), x.norm(dim=-1))


def test_boundary_modes_are_explicit() -> None:
    connection = torch.tensor([[[0.1, 0.2]]])
    with pytest.raises(ValueError, match="exceeds max_positions"):
        connection_to_absolute_phases(connection, 5)

    repeated = connection_to_absolute_phases(connection, 5, boundary="repeat_last")
    frozen = connection_to_absolute_phases(connection, 5, boundary="zero")
    torch.testing.assert_close(repeated.flatten(), torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7]))
    torch.testing.assert_close(frozen.flatten(), torch.tensor([0.0, 0.1, 0.3, 0.3, 0.3]))


def test_helpers_validate_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        connection_to_absolute_phases(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="positive even"):
        rope_frequencies(3)
    with pytest.raises(ValueError, match="one angle"):
        rotate_pairs(torch.zeros(1, 8), torch.zeros(1, 3))
