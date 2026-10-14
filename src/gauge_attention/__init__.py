"""Gauge-inspired learned rotary transport for PyTorch attention."""

from .attention import GaugeAttention
from .functional import (
    connection_to_absolute_phases,
    pairwise_transport_phases,
    rope_connection,
    rope_frequencies,
    rotate_pairs,
)

__all__ = [
    "GaugeAttention",
    "connection_to_absolute_phases",
    "pairwise_transport_phases",
    "rope_connection",
    "rope_frequencies",
    "rotate_pairs",
]

__version__ = "0.1.0"
