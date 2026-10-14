"""Small, inspectable functions for discrete rotary transport."""

from __future__ import annotations

from typing import Literal

import torch

BoundaryMode = Literal["error", "repeat_last", "zero"]


def _validate_head_dim(head_dim: int) -> None:
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be a positive even number, got {head_dim}")


def rope_frequencies(
    head_dim: int,
    *,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the standard RoPE angular increment for each coordinate pair."""
    _validate_head_dim(head_dim)
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    dtype = dtype or torch.get_default_dtype()
    pair_indices = torch.arange(head_dim // 2, device=device, dtype=dtype)
    return base ** (-2 * pair_indices / head_dim)


def rope_connection(
    n_heads: int,
    head_dim: int,
    max_positions: int,
    *,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a constant connection whose accumulated phases are standard RoPE phases.

    The final dimension stores edges: entry ``k`` is the phase increment from
    position ``k`` to position ``k + 1``. There are therefore
    ``max_positions - 1`` entries.
    """
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    if max_positions < 2:
        raise ValueError(f"max_positions must be at least 2, got {max_positions}")
    frequencies = rope_frequencies(head_dim, base=base, device=device, dtype=dtype)
    return frequencies.view(1, -1, 1).expand(n_heads, -1, max_positions - 1).clone()


def _connection_for_length(
    connection: torch.Tensor,
    seq_len: int,
    boundary: BoundaryMode,
) -> torch.Tensor:
    if connection.ndim != 3:
        raise ValueError(
            "connection must have shape (heads, coordinate_pairs, edges), "
            f"got {tuple(connection.shape)}"
        )
    if connection.shape[-1] < 1:
        raise ValueError("connection must contain at least one edge")
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if boundary not in {"error", "repeat_last", "zero"}:
        raise ValueError(f"unknown boundary mode: {boundary!r}")

    edges_needed = seq_len - 1
    learned_edges = connection.shape[-1]
    if edges_needed <= learned_edges:
        return connection[..., :edges_needed]
    if boundary == "error":
        max_positions = learned_edges + 1
        raise ValueError(
            f"seq_len={seq_len} exceeds max_positions={max_positions}; "
            "choose boundary='repeat_last' or boundary='zero' explicitly"
        )

    extra_edges = edges_needed - learned_edges
    if boundary == "repeat_last":
        extension = connection[..., -1:].expand(*connection.shape[:-1], extra_edges)
    else:
        extension = connection.new_zeros(*connection.shape[:-1], extra_edges)
    return torch.cat((connection, extension), dim=-1)


def connection_to_absolute_phases(
    connection: torch.Tensor,
    seq_len: int | None = None,
    *,
    boundary: BoundaryMode = "error",
) -> torch.Tensor:
    """Accumulate edge values into a phase at each position.

    For connection ``A``, the result satisfies ``theta[0] = 0`` and
    ``theta[i] = sum(A[k] for k in range(i))``. The shape is
    ``(heads, seq_len, coordinate_pairs)``.

    Requests past the learned boundary are rejected by default. The caller can
    instead explicitly repeat the last edge value or append zero increments.
    """
    if seq_len is None:
        if connection.ndim != 3:
            raise ValueError(
                "connection must have shape (heads, coordinate_pairs, edges), "
                f"got {tuple(connection.shape)}"
            )
        seq_len = connection.shape[-1] + 1
    used_connection = _connection_for_length(connection, seq_len, boundary)
    origin = connection.new_zeros(connection.shape[0], connection.shape[1], 1)
    phases = torch.cat((origin, torch.cumsum(used_connection, dim=-1)), dim=-1)
    return phases.permute(0, 2, 1)


def pairwise_transport_phases(absolute_phases: torch.Tensor) -> torch.Tensor:
    """Return transport phases from every query position to every key position.

    If ``theta`` has shape ``(heads, positions, coordinate_pairs)``, the result
    has shape ``(heads, query_positions, key_positions, coordinate_pairs)`` and
    contains ``theta[key] - theta[query]``.
    """
    if absolute_phases.ndim != 3:
        raise ValueError(
            "absolute_phases must have shape (heads, positions, coordinate_pairs), "
            f"got {tuple(absolute_phases.shape)}"
        )
    query_phases = absolute_phases[:, :, None, :]
    key_phases = absolute_phases[:, None, :, :]
    return key_phases - query_phases


def rotate_pairs(x: torch.Tensor, pair_phases: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of ``x`` using one angle per coordinate pair.

    The final dimension of ``x`` is interpreted as ``[x_1, x_2]`` with equal
    halves. ``pair_phases`` must broadcast to ``x_1``.
    """
    _validate_head_dim(x.shape[-1])
    first, second = x.chunk(2, dim=-1)
    if pair_phases.shape[-1] != first.shape[-1]:
        raise ValueError(
            "pair_phases must have one angle per coordinate pair: "
            f"expected {first.shape[-1]}, got {pair_phases.shape[-1]}"
        )
    cosine = pair_phases.cos()
    sine = pair_phases.sin()
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine),
        dim=-1,
    )
