from __future__ import annotations

import io

import pytest
import torch

from gauge_attention import GaugeAttention, NoPositionAttention, RoPEAttention


def _copy_projections(source: torch.nn.Module, target: torch.nn.Module) -> None:
    names = ("q_proj", "k_proj", "v_proj", "out_proj")
    with torch.no_grad():
        for name in names:
            getattr(target, name).load_state_dict(getattr(source, name).state_dict())


def test_zero_connection_matches_position_free_scores_and_outputs() -> None:
    torch.manual_seed(2)
    gauge = GaugeAttention(16, 2, 8, connection_init="zeros").eval()
    plain = NoPositionAttention(16, 2).eval()
    _copy_projections(gauge, plain)
    x = torch.randn(3, 6, 16)

    torch.testing.assert_close(gauge.attention_logits(x), plain.attention_logits(x))
    torch.testing.assert_close(gauge(x), plain(x))


def test_constant_rope_connection_matches_rope_scores_and_outputs() -> None:
    torch.manual_seed(3)
    gauge = GaugeAttention(24, 3, 12, connection_init="rope").eval()
    rope = RoPEAttention(24, 3).eval()
    _copy_projections(gauge, rope)
    x = torch.randn(2, 9, 24)

    torch.testing.assert_close(gauge.attention_logits(x), rope.attention_logits(x))
    torch.testing.assert_close(gauge(x), rope(x))


def test_connection_and_input_receive_gradients() -> None:
    torch.manual_seed(4)
    layer = GaugeAttention(16, 2, 6)
    x = torch.randn(2, 6, 16, requires_grad=True)
    loss = layer(x, is_causal=True).square().mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.connection.grad is not None
    assert torch.isfinite(layer.connection.grad).all()
    assert layer.connection.grad.abs().sum() > 0


def test_parameter_updates_are_used_without_a_stale_cache() -> None:
    torch.manual_seed(8)
    layer = GaugeAttention(16, 2, 6).eval()
    x = torch.randn(1, 6, 16)
    logits_before = layer.attention_logits(x)
    with torch.no_grad():
        layer.connection.fill_(0.3)
    logits_after = layer.attention_logits(x)
    assert not torch.allclose(logits_before, logits_after)


@pytest.mark.parametrize("mask_shape", [(5, 5), (2, 5, 5), (2, 1, 5, 5), (2, 2, 5, 5)])
def test_boolean_mask_shapes(mask_shape: tuple[int, ...]) -> None:
    layer = GaugeAttention(16, 2, 5)
    x = torch.randn(2, 5, 16)
    mask = torch.ones(mask_shape, dtype=torch.bool).tril()
    assert layer(x, mask).shape == x.shape


def test_boolean_and_additive_masks_match() -> None:
    layer = GaugeAttention(16, 2, 5).eval()
    x = torch.randn(2, 5, 16)
    allowed = torch.ones(5, 5, dtype=torch.bool).tril()
    additive = torch.zeros(5, 5).masked_fill(~allowed, float("-inf"))
    torch.testing.assert_close(layer(x, allowed), layer(x, additive))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_and_dtype(dtype: torch.dtype) -> None:
    layer = GaugeAttention(16, 2, 7).to(dtype=dtype)
    x = torch.randn(2, 7, 16, dtype=dtype)
    output = layer(x, is_causal=True)
    assert output.shape == x.shape
    assert output.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_device() -> None:
    layer = GaugeAttention(16, 2, 7).cuda()
    x = torch.randn(2, 7, 16, device="cuda")
    assert layer(x).device.type == "cuda"


def test_state_dict_round_trip() -> None:
    torch.manual_seed(5)
    original = GaugeAttention(16, 2, 6, boundary="repeat_last").eval()
    with torch.no_grad():
        original.connection.normal_()
    x = torch.randn(1, 8, 16)

    payload = io.BytesIO()
    torch.save(original.state_dict(), payload)
    payload.seek(0)
    restored = GaugeAttention(16, 2, 6, boundary="repeat_last").eval()
    restored.load_state_dict(torch.load(payload, weights_only=True))

    torch.testing.assert_close(restored.connection, original.connection)
    torch.testing.assert_close(restored(x), original(x))


def test_beyond_max_errors_by_default_and_obeys_boundary_mode() -> None:
    x = torch.randn(1, 6, 8)
    strict = GaugeAttention(8, 1, 4)
    with pytest.raises(ValueError, match="exceeds max_positions"):
        strict(x)

    repeated = GaugeAttention(8, 1, 4, boundary="repeat_last")
    frozen = GaugeAttention(8, 1, 4, boundary="zero")
    assert repeated(x).shape == x.shape
    assert frozen(x).shape == x.shape


def test_rotate_values_is_disabled_by_default() -> None:
    assert GaugeAttention(8, 1, 4).rotate_values is False


def test_masks_are_validated() -> None:
    layer = GaugeAttention(8, 1, 4)
    x = torch.randn(2, 4, 8)
    with pytest.raises(ValueError, match="last dimensions"):
        layer(x, torch.ones(3, 3, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        layer(x, torch.ones(4, 4, dtype=torch.int64))
    with pytest.raises(ValueError, match="at least one finite"):
        layer(x, torch.zeros(4, 4, dtype=torch.bool))


def test_causal_and_boolean_mask_cannot_hide_every_key() -> None:
    layer = GaugeAttention(8, 1, 4)
    x = torch.randn(1, 4, 8)
    future_only = torch.ones(4, 4, dtype=torch.bool).triu(1)
    with pytest.raises(ValueError, match="at least one finite"):
        layer(x, future_only, is_causal=True)


def test_optional_value_transport_runs_and_has_gradients() -> None:
    layer = GaugeAttention(8, 1, 4, rotate_values=True)
    x = torch.randn(2, 4, 8, requires_grad=True)
    output = layer(x, is_causal=True)
    output.sum().backward()
    assert output.shape == x.shape
    assert layer.connection.grad is not None
