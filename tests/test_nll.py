import os

import pytest
import torch
import torch.nn.functional as F

from fused_cross_entropy import TritonFusedNLL

# Configurations to test (N1, N2, N3)
CONFIGS = [
    (1, 1, 1),
    (2, 2, 2),
    (3, 3, 3),
    (13, 13, 13),
    (32, 64, 128),
    (128, 256, 512),
    (33, 65, 129),
    (1, 16, 16),
    (64, 32, 64),
    (127, 127, 127),
    (10, 100, 50),
    (50, 10, 100),
    (16, 1024, 16),
    (16, 16, 1024),
    (1024, 16, 16),
    (42, 42, 42),
    (200, 200, 200),
]


@pytest.mark.parametrize("N, D_in, D_out", CONFIGS)
@pytest.mark.parametrize("has_bias", [True, False])
def test_fused_nll_accuracy(N, D_in, D_out, has_bias):
    """
    Tests the numerical accuracy of the Triton Fused NLL kernel against
    the PyTorch reference implementation.
    This test is designed to run on CPU via the Triton interpreter.
    """
    # The test must run on CPU for CI environments.
    # The TRITON_INTERPRET=1 env var forces the interpreter backend.
    assert os.getenv("TRITON_INTERPRET") == "1", (
        "This test must be run with TRITON_INTERPRET=1"
    )

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    x = torch.randn(N, D_in, device=device, requires_grad=True, dtype=dtype)
    target = torch.randint(0, D_out, (N,), device=device)
    weight = torch.randn(D_out, D_in, device=device, requires_grad=True, dtype=dtype)
    bias = (
        torch.randn(D_out, device=device, requires_grad=True, dtype=dtype)
        if has_bias
        else None
    )

    # --- Triton Implementation ---
    loss_triton = TritonFusedNLL.apply(x, target, weight, bias)
    loss_triton.sum().backward()
    grad_x_triton = x.grad.clone()
    grad_w_triton = weight.grad.clone()
    grad_b_triton = bias.grad.clone() if has_bias else None

    x.grad.zero_()
    weight.grad.zero_()
    if has_bias:
        bias.grad.zero_()

    # --- PyTorch Reference Implementation ---
    logits = F.linear(x, weight, bias)
    loss_ref = F.cross_entropy(logits, target, reduction="none")
    loss_ref.sum().backward()

    # --- Numerical Validation ---
    assert torch.allclose(loss_triton, loss_ref, atol=1.0, rtol=0)
    assert torch.allclose(grad_x_triton, x.grad, atol=0.1, rtol=0)
    assert torch.allclose(grad_w_triton, weight.grad, atol=0.1, rtol=0)
    if has_bias:
        assert torch.allclose(grad_b_triton, bias.grad, atol=0.1, rtol=0)
