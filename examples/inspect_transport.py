"""Inspect a tiny learned connection and run a forward pass."""

import torch

from gauge_attention import GaugeAttention

torch.manual_seed(7)

attention = GaugeAttention(
    embed_dim=8,
    n_heads=2,
    max_positions=4,
    connection_init="zeros",
)

with torch.no_grad():
    # Edge k connects position k to k + 1. Here both coordinate pairs in each
    # head use the same three illustrative increments.
    attention.connection.copy_(torch.tensor([0.10, -0.05, 0.20]).view(1, 1, 3).expand(2, 2, 3))

print("absolute phases (head 0):")
print(attention.absolute_phases()[0])
print("transport phases from query 1 to every key (head 0):")
print(attention.transport_phases()[0, 1])

x = torch.randn(1, 4, 8)
y = attention(x, is_causal=True)
print("output shape:", tuple(y.shape))
