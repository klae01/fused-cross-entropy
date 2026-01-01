import torch
import triton
import triton.language as tl


@triton.jit
def _fused_nll_fwd_kernel(
    X,
    Y,
    W,
    B,
    Out,
    LSE,
    stride_x_n1,
    stride_x_n2,
    stride_w_n3,
    stride_w_n2,
    stride_b_n3,
    N1: tl.constexpr,
    N2: tl.constexpr,
    N3: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_N3: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n1 = pid * BLOCK_N1 + tl.arange(0, BLOCK_N1)
    mask_n1 = offs_n1 < N1
    target = tl.load(Y + offs_n1, mask=mask_n1, other=0)
    m_i = tl.zeros([BLOCK_N1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_N1], dtype=tl.float32)
    target_logit = tl.zeros([BLOCK_N1], dtype=tl.float32)
    offs_n2, offs_n3 = tl.arange(0, BLOCK_N2), tl.arange(0, BLOCK_N3)
    for off_n3_start in range(0, N3, BLOCK_N3):
        offs_n3_curr = off_n3_start + offs_n3
        mask_n3 = offs_n3_curr < N3
        acc = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(B + offs_n3_curr * stride_b_n3, mask=mask_n3, other=0.0).to(
                tl.float32
            )[None, :]
        for off_n2_start in range(0, N2, BLOCK_N2):
            offs_n2_curr = off_n2_start + offs_n2
            mask_n2 = offs_n2_curr < N2
            x_chunk = tl.load(
                X
                + (
                    offs_n1[:, None] * stride_x_n1 + offs_n2_curr[None, :] * stride_x_n2
                ),
                mask=mask_n1[:, None] & mask_n2[None, :],
                other=0.0,
            )
            w_chunk = tl.load(
                W
                + (
                    offs_n3_curr[None, :] * stride_w_n3
                    + offs_n2_curr[:, None] * stride_w_n2
                ),
                mask=mask_n2[:, None] & mask_n3[None, :],
                other=0.0,
            )
            acc = tl.dot(x_chunk, w_chunk, acc, allow_tf32=True)
        acc = tl.where(mask_n3[None, :], acc, -float("inf"))
        m_curr = tl.max(acc, 1)
        m_new = tl.maximum(m_i, m_curr)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(acc - m_new[:, None]), 1)
        m_i = m_new
        is_target = offs_n3_curr[None, :] == target[:, None]
        target_logit += tl.sum(
            tl.where(is_target & mask_n1[:, None] & mask_n3[None, :], acc, 0.0), 1
        )
    lse = m_i + tl.log(l_i)
    tl.store(LSE + offs_n1, lse, mask=mask_n1)

    loss = (lse - target_logit).to(X.dtype.element_ty)
    tl.store(Out + offs_n1, loss, mask=mask_n1)


@triton.jit
def _fused_nll_bwd_dx_kernel(
    X,
    Y,
    GradOut,
    W,
    B,
    GradX,
    LSE,
    stride_x_n1,
    stride_x_n2,
    stride_w_n3,
    stride_w_n2,
    stride_b_n3,
    N1: tl.constexpr,
    N2: tl.constexpr,
    N3: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_N3: tl.constexpr,
):
    pid_n1 = tl.program_id(0)
    offs_n1 = pid_n1 * BLOCK_N1 + tl.arange(0, BLOCK_N1)
    mask_n1 = offs_n1 < N1
    lse = tl.load(LSE + offs_n1, mask=mask_n1, other=0.0)
    target = tl.load(Y + offs_n1, mask=mask_n1, other=0)
    grad_out = tl.load(GradOut + offs_n1, mask=mask_n1, other=0.0).to(tl.float32)

    offs_n2, offs_n3 = tl.arange(0, BLOCK_N2), tl.arange(0, BLOCK_N3)

    for off_n3_start in range(0, N3, BLOCK_N3):
        offs_n3_curr = off_n3_start + offs_n3
        mask_n3 = offs_n3_curr < N3
        # Recompute logits
        logits = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=tl.float32)
        if HAS_BIAS:
            logits += tl.load(
                B + offs_n3_curr * stride_b_n3, mask=mask_n3, other=0.0
            ).to(tl.float32)[None, :]
        for off_n2_start in range(0, N2, BLOCK_N2):
            offs_n2_curr = off_n2_start + offs_n2
            mask_n2 = offs_n2_curr < N2
            x_chunk = tl.load(
                X
                + (
                    offs_n1[:, None] * stride_x_n1 + offs_n2_curr[None, :] * stride_x_n2
                ),
                mask=mask_n1[:, None] & mask_n2[None, :],
                other=0.0,
            )
            w_chunk = tl.load(
                W
                + (
                    offs_n3_curr[None, :] * stride_w_n3
                    + offs_n2_curr[:, None] * stride_w_n2
                ),
                mask=mask_n2[:, None] & mask_n3[None, :],
                other=0.0,
            )
            logits = tl.dot(x_chunk, w_chunk, logits, allow_tf32=True)

        logits = tl.where(mask_n1[:, None] & mask_n3[None, :], logits, -float("inf"))
        probs = tl.exp(logits - lse[:, None])
        probs = tl.where(mask_n1[:, None] & mask_n3[None, :], probs, 0.0)
        is_target = offs_n3_curr[None, :] == target[:, None]
        probs = tl.where(is_target & mask_n1[:, None], probs - 1.0, probs)
        grad_logits = (probs * grad_out[:, None]).to(X.dtype.element_ty)

        for off_n2_start in range(0, N2, BLOCK_N2):
            offs_n2_curr = off_n2_start + offs_n2
            mask_n2 = offs_n2_curr < N2
            w_chunk_t = tl.load(
                W
                + (
                    offs_n3_curr[:, None] * stride_w_n3
                    + offs_n2_curr[None, :] * stride_w_n2
                ),
                mask=mask_n3[:, None] & mask_n2[None, :],
                other=0.0,
            )
            dx_part = tl.dot(grad_logits, w_chunk_t, allow_tf32=True)

            # Use add store
            grad_x_ptr = GradX + (
                offs_n1[:, None] * stride_x_n1 + offs_n2_curr[None, :] * stride_x_n2
            )
            current_grad_x = tl.load(
                grad_x_ptr, mask=mask_n1[:, None] & mask_n2[None, :], other=0.0
            )
            tl.store(
                grad_x_ptr,
                current_grad_x + dx_part.to(X.dtype.element_ty),
                mask=mask_n1[:, None] & mask_n2[None, :],
            )


@triton.jit
def _fused_nll_bwd_dw_kernel(
    X,
    Y,
    GradOut,
    W,
    B,
    GradW,
    LSE,
    stride_x_n1,
    stride_x_n2,
    stride_w_n3,
    stride_w_n2,
    stride_b_n3,
    N1: tl.constexpr,
    N2: tl.constexpr,
    N3: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_N3: tl.constexpr,
):
    pid_n3 = tl.program_id(0)
    offs_n3 = pid_n3 * BLOCK_N3 + tl.arange(0, BLOCK_N3)
    mask_n3 = offs_n3 < N3

    offs_n2, offs_n1 = tl.arange(0, BLOCK_N2), tl.arange(0, BLOCK_N1)

    for off_n1_start in range(0, N1, BLOCK_N1):
        offs_n1_curr = off_n1_start + offs_n1
        mask_n1 = offs_n1_curr < N1
        lse = tl.load(LSE + offs_n1_curr, mask=mask_n1, other=0.0)
        target = tl.load(Y + offs_n1_curr, mask=mask_n1, other=0)
        grad_out = tl.load(GradOut + offs_n1_curr, mask=mask_n1, other=0.0).to(
            tl.float32
        )

        # Recompute logits
        logits = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=tl.float32)
        if HAS_BIAS:
            logits += tl.load(B + offs_n3 * stride_b_n3, mask=mask_n3, other=0.0).to(
                tl.float32
            )[None, :]

        for off_n2_start in range(0, N2, BLOCK_N2):
            offs_n2_curr = off_n2_start + offs_n2
            mask_n2 = offs_n2_curr < N2
            x_chunk = tl.load(
                X
                + (
                    offs_n1_curr[:, None] * stride_x_n1
                    + offs_n2_curr[None, :] * stride_x_n2
                ),
                mask=mask_n1[:, None] & mask_n2[None, :],
                other=0.0,
            )
            w_chunk = tl.load(
                W
                + (
                    offs_n3[None, :] * stride_w_n3 + offs_n2_curr[:, None] * stride_w_n2
                ),
                mask=mask_n2[:, None] & mask_n3[None, :],
                other=0.0,
            )
            logits = tl.dot(x_chunk, w_chunk, logits, allow_tf32=True)

        logits = tl.where(mask_n1[:, None] & mask_n3[None, :], logits, -float("inf"))
        probs = tl.exp(logits - lse[:, None])
        probs = tl.where(mask_n1[:, None] & mask_n3[None, :], probs, 0.0)
        is_target = offs_n3[None, :] == target[:, None]
        probs = tl.where(is_target & mask_n1[:, None], probs - 1.0, probs)
        grad_logits = (probs * grad_out[:, None]).to(X.dtype.element_ty)

        for off_n2_start in range(0, N2, BLOCK_N2):
            offs_n2_curr = off_n2_start + offs_n2
            mask_n2 = offs_n2_curr < N2
            x_chunk = tl.load(
                X
                + (
                    offs_n1_curr[:, None] * stride_x_n1
                    + offs_n2_curr[None, :] * stride_x_n2
                ),
                mask=mask_n1[:, None] & mask_n2[None, :],
                other=0.0,
            )
            dw_part = tl.dot(tl.trans(grad_logits), x_chunk, allow_tf32=True)

            # Use add store
            grad_w_ptr = GradW + (
                offs_n3[:, None] * stride_w_n3 + offs_n2_curr[None, :] * stride_w_n2
            )
            current_grad_w = tl.load(
                grad_w_ptr, mask=mask_n3[:, None] & mask_n2[None, :], other=0.0
            )
            tl.store(
                grad_w_ptr,
                current_grad_w + dw_part.to(W.dtype.element_ty),
                mask=mask_n3[:, None] & mask_n2[None, :],
            )


@triton.jit
def _fused_nll_bwd_db_kernel(
    X,
    Y,
    GradOut,
    W,
    B,
    GradB,
    LSE,
    stride_x_n1,
    stride_x_n2,
    stride_w_n3,
    stride_w_n2,
    stride_b_n3,
    N1: tl.constexpr,
    N2: tl.constexpr,
    N3: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_N3: tl.constexpr,
):
    pid_n3 = tl.program_id(0)
    offs_n3 = pid_n3 * BLOCK_N3 + tl.arange(0, BLOCK_N3)
    mask_n3 = offs_n3 < N3
    db_acc = tl.zeros([BLOCK_N3], dtype=tl.float32)
    offs_n1, offs_n2 = tl.arange(0, BLOCK_N1), tl.arange(0, BLOCK_N2)
    for off_n1_start in range(0, N1, BLOCK_N1):
        offs_n1_curr = off_n1_start + offs_n1
        mask_n1 = offs_n1_curr < N1
        target, lse, grad_out = (
            tl.load(Y + offs_n1_curr, mask=mask_n1, other=0),
            tl.load(LSE + offs_n1_curr, mask=mask_n1, other=0.0),
            tl.load(GradOut + offs_n1_curr, mask=mask_n1, other=0.0).to(tl.float32),
        )
        logits = (
            tl.zeros([BLOCK_N1, BLOCK_N3], dtype=tl.float32)
            + tl.load(B + offs_n3 * stride_b_n3, mask=mask_n3, other=0.0).to(
                tl.float32
            )[None, :]
        )
        for off_n2_start in range(0, N2, BLOCK_N2):
            offs_n2_curr = off_n2_start + offs_n2
            mask_n2 = offs_n2_curr < N2
            x_chunk = tl.load(
                X
                + (
                    offs_n1_curr[:, None] * stride_x_n1
                    + offs_n2_curr[None, :] * stride_x_n2
                ),
                mask=mask_n1[:, None] & mask_n2[None, :],
                other=0.0,
            )
            w_chunk = tl.load(
                W
                + (
                    offs_n3[None, :] * stride_w_n3 + offs_n2_curr[:, None] * stride_w_n2
                ),
                mask=mask_n2[:, None] & mask_n3[None, :],
                other=0.0,
            )
            logits = tl.dot(x_chunk, w_chunk, logits, allow_tf32=True)
        logits = tl.where(mask_n1[:, None] & mask_n3[None, :], logits, -float("inf"))
        probs = tl.exp(logits - lse[:, None])
        probs = tl.where(mask_n1[:, None] & mask_n3[None, :], probs, 0.0)
        is_target = offs_n3[None, :] == target[:, None]
        probs = tl.where(is_target & mask_n1[:, None], probs - 1.0, probs)
        db_acc += tl.sum(probs * grad_out[:, None], 0)

    tl.store(GradB + offs_n3 * stride_b_n3, db_acc.to(B.dtype.element_ty), mask=mask_n3)


class TritonFusedNLL(torch.autograd.Function):
    TRITON_KWARGS = dict(
        BLOCK_N1=64,
        BLOCK_N2=64,
        BLOCK_N3=64,
        num_warps=4,
    )

    @staticmethod
    def forward(ctx, x, target, weight, bias=None):
        target_i32 = target.to(torch.int32).contiguous()
        has_bias = bias is not None
        if not has_bias:
            bias = x.new_empty((1,), dtype=x.dtype)
        N1, N2, N3 = x.shape[0], x.shape[1], weight.shape[0]
        out = torch.empty((N1,), device=x.device, dtype=x.dtype)
        lse = torch.empty((N1,), device=x.device, dtype=torch.float32)
        _fused_nll_fwd_kernel[lambda META: (triton.cdiv(N1, META["BLOCK_N1"]),)](
            x,
            target_i32,
            weight,
            bias,
            out,
            lse,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            bias.stride(0),
            N1,
            N2,
            N3,
            HAS_BIAS=has_bias,
            **TritonFusedNLL.TRITON_KWARGS,
        )
        ctx.save_for_backward(x, target_i32, weight, bias, lse)
        ctx.has_bias = has_bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, target_i32, weight, bias, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        N1, N2, N3 = x.shape[0], x.shape[1], weight.shape[0]
        grad_x = torch.zeros_like(x)
        grad_w = torch.zeros_like(weight)
        grad_b = torch.zeros_like(bias) if ctx.has_bias else None
        _fused_nll_bwd_dx_kernel[lambda META: (triton.cdiv(N1, META["BLOCK_N1"]),)](
            x,
            target_i32,
            grad_output,
            weight,
            bias,
            grad_x,
            lse,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            bias.stride(0),
            N1,
            N2,
            N3,
            HAS_BIAS=ctx.has_bias,
            **TritonFusedNLL.TRITON_KWARGS,
        )
        _fused_nll_bwd_dw_kernel[lambda META: (triton.cdiv(N3, META["BLOCK_N3"]),)](
            x,
            target_i32,
            grad_output,
            weight,
            bias,
            grad_w,
            lse,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            bias.stride(0),
            N1,
            N2,
            N3,
            HAS_BIAS=ctx.has_bias,
            **TritonFusedNLL.TRITON_KWARGS,
        )
        if ctx.has_bias:
            _fused_nll_bwd_db_kernel[lambda META: (triton.cdiv(N3, META["BLOCK_N3"]),)](
                x,
                target_i32,
                grad_output,
                weight,
                bias,
                grad_b,
                lse,
                x.stride(0),
                x.stride(1),
                weight.stride(0),
                weight.stride(1),
                bias.stride(0),
                N1,
                N2,
                N3,
                **TritonFusedNLL.TRITON_KWARGS,
            )
        return grad_x, None, grad_w, grad_b


def triton_fused_nll(x, target, weight, bias=None):
    return TritonFusedNLL.apply(x, target, weight, bias)
