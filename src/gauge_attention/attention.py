"""Reference attention layers built around rotary transport."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from .functional import (
    BoundaryMode,
    connection_to_absolute_phases,
    pairwise_transport_phases,
    rope_connection,
    rotate_pairs,
)


class _ProjectedAttention(nn.Module):
    """Shared projections and explicit scaled dot-product attention."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or n_heads <= 0 or embed_dim % n_heads:
            raise ValueError("embed_dim must be positive and divisible by n_heads")
        head_dim = embed_dim // n_heads
        if head_dim % 2:
            raise ValueError("the dimension of each attention head must be even")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, positions, _ = x.shape
        return x.reshape(batch, positions, self.n_heads, self.head_dim).transpose(1, 2)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"x must have shape (batch, positions, {self.embed_dim}), got {tuple(x.shape)}"
            )
        return (
            self._split_heads(self.q_proj(x)),
            self._split_heads(self.k_proj(x)),
            self._split_heads(self.v_proj(x)),
        )

    def _validate_mask(self, mask: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch, heads, queries, keys = scores.shape
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        elif mask.ndim != 4:
            raise ValueError("mask must have 2, 3, or 4 dimensions")
        expected_tail = (queries, keys)
        if tuple(mask.shape[-2:]) != expected_tail:
            raise ValueError(
                f"mask's last dimensions must be {expected_tail}, got {tuple(mask.shape[-2:])}"
            )
        if mask.shape[0] not in {1, batch} or mask.shape[1] not in {1, heads}:
            raise ValueError(
                "mask batch/head dimensions must be 1 or match the input; "
                f"got {tuple(mask.shape[:2])}, expected ({batch}, {heads})"
            )
        if mask.device != scores.device:
            raise ValueError("mask and input must be on the same device")
        return mask

    def _masked_scores(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        if is_causal:
            causal_mask = torch.ones(
                scores.shape[-2:], device=scores.device, dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if mask is not None:
            mask = self._validate_mask(mask, scores)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            elif torch.is_floating_point(mask):
                scores = scores + mask.to(dtype=scores.dtype)
            else:
                raise TypeError(
                    "mask must be boolean (True means allowed) or floating-point additive"
                )
        if bool(torch.isneginf(scores).all(dim=-1).any()):
            raise ValueError("every mask row must leave at least one finite attention logit")
        return scores

    def _logits_from_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        return self._masked_scores(scores, mask, is_causal)

    def _finish(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        logits = self._logits_from_qk(q, k, mask=mask, is_causal=is_causal)
        weights = self.attention_dropout(logits.softmax(dim=-1))
        output = weights @ v
        batch, _, positions, _ = output.shape
        output = output.transpose(1, 2).contiguous().reshape(batch, positions, self.embed_dim)
        return self.output_dropout(self.out_proj(output))

    def _position_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return q, k, v

    def attention_logits(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Return pre-softmax attention logits for inspection and testing."""
        q, k, v = self._project(x)
        q, k, _ = self._position_qkv(q, k, v)
        return self._logits_from_qk(q, k, mask=mask, is_causal=is_causal)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        q, k, v = self._project(x)
        q, k, v = self._position_qkv(q, k, v)
        return self._finish(q, k, v, mask=mask, is_causal=is_causal)


class GaugeAttention(_ProjectedAttention):
    """Attention with learned, position-dependent rotary edge increments.

    Each parameter ``connection[h, p, k]`` is the angle added while moving from
    position ``k`` to ``k + 1`` for attention head ``h`` and coordinate pair
    ``p``. Accumulating these increments gives an absolute frame at each
    position. Rotating queries and keys into those frames makes their dot
    product depend on the transport phase between the two positions.

    Unlike standard RoPE, a nonconstant connection is position-dependent: two
    pairs of tokens at the same distance can receive different transport
    phases. Requests beyond the learned boundary follow ``boundary`` explicitly.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        max_positions: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
        connection_init: Literal["zeros", "rope"] = "zeros",
        rope_base: float = 10_000.0,
        boundary: BoundaryMode = "error",
        rotate_values: bool = False,
    ) -> None:
        super().__init__(embed_dim, n_heads, dropout=dropout, bias=bias)
        if max_positions < 2:
            raise ValueError("max_positions must be at least 2")
        if boundary not in {"error", "repeat_last", "zero"}:
            raise ValueError(f"unknown boundary mode: {boundary!r}")
        self.max_positions = max_positions
        self.boundary = boundary
        self.rotate_values = rotate_values

        if connection_init == "zeros":
            initial_connection = torch.zeros(n_heads, self.head_dim // 2, max_positions - 1)
        elif connection_init == "rope":
            initial_connection = rope_connection(
                n_heads, self.head_dim, max_positions, base=rope_base
            )
        else:
            raise ValueError(f"unknown connection_init: {connection_init!r}")
        self.connection = nn.Parameter(initial_connection)

    def absolute_phases(self, seq_len: int | None = None) -> torch.Tensor:
        """Return accumulated phases, preserving gradients to the connection."""
        return connection_to_absolute_phases(self.connection, seq_len, boundary=self.boundary)

    def transport_phases(self, seq_len: int | None = None) -> torch.Tensor:
        """Return ``theta[key] - theta[query]`` for all position pairs."""
        return pairwise_transport_phases(self.absolute_phases(seq_len))

    def _position_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phases = self.absolute_phases(q.shape[-2]).to(dtype=q.dtype)[None, ...]
        q = rotate_pairs(q, phases)
        k = rotate_pairs(k, phases)
        if self.rotate_values:
            v = rotate_pairs(v, phases)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        q, k, v = self._project(x)
        phases = self.absolute_phases(q.shape[-2]).to(dtype=q.dtype)[None, ...]
        q = rotate_pairs(q, phases)
        k = rotate_pairs(k, phases)
        if self.rotate_values:
            v = rotate_pairs(v, phases)
        if not self.rotate_values:
            return self._finish(q, k, v, mask=mask, is_causal=is_causal)

        # Inverse transport must happen before heads are merged and projected.
        logits = self._logits_from_qk(q, k, mask=mask, is_causal=is_causal)
        weights = self.attention_dropout(logits.softmax(dim=-1))
        transported = rotate_pairs(weights @ v, -phases)
        batch, _, positions, _ = transported.shape
        transported = (
            transported.transpose(1, 2).contiguous().reshape(batch, positions, self.embed_dim)
        )
        return self.output_dropout(self.out_proj(transported))
