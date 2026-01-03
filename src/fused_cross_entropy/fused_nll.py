import torch
import triton
import triton.language as tl

p_dtype = tl.float32
TORCH_P_DTYPE = torch.float32


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
    N1,
    N2: tl.constexpr,
    N3,
    HAS_BIAS: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_N3: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
):
    c_dtype = X.dtype.element_ty
    pid = tl.program_id(0)
    offs_n1 = pid * BLOCK_N1 + tl.arange(0, BLOCK_N1)
    mask_n1 = offs_n1 < N1
    target_y = tl.load(Y + offs_n1, mask=mask_n1, other=0)
    m_i = tl.zeros([BLOCK_N1], dtype=p_dtype) - float("inf")
    l_i = tl.zeros([BLOCK_N1], dtype=p_dtype)
    target_logit = tl.zeros([BLOCK_N1], dtype=p_dtype)
    x_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N1, N2),
        strides=(stride_x_n1, stride_x_n2),
        offsets=(pid * BLOCK_N1, 0),
        block_shape=(BLOCK_N1, BLOCK_N2),
        order=(1, 0),
    )
    offs_n3 = tl.arange(0, BLOCK_N3)
    off_n3_start = 0
    while off_n3_start < N3:
        offs_n3_curr = off_n3_start + offs_n3
        mask_n3 = offs_n3_curr < N3
        logits = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=p_dtype)
        if HAS_BIAS:
            bias_val = tl.load(B + offs_n3_curr * stride_b_n3, mask=mask_n3, other=0.0)
            logits += bias_val.to(p_dtype)[None, :]
        w_block_ptr = tl.make_block_ptr(
            base=W,
            shape=(N3, N2),
            strides=(stride_w_n3, stride_w_n2),
            offsets=(off_n3_start, 0),
            block_shape=(BLOCK_N3, BLOCK_N2),
            order=(1, 0),
        )
        x_ptr = x_block_ptr
        w_ptr = w_block_ptr
        for k in range(0, N2, BLOCK_N2):
            x_chunk = tl.load(x_ptr, boundary_check=(0, 1))
            w_chunk = tl.load(w_ptr, boundary_check=(0, 1))
            logits = tl.dot(
                x_chunk,
                tl.trans(w_chunk),
                logits,
                input_precision=INPUT_PRECISION,
                out_dtype=p_dtype,
            )
            x_ptr = tl.advance(x_ptr, (0, BLOCK_N2))
            w_ptr = tl.advance(w_ptr, (0, BLOCK_N2))
        logits = tl.where(mask_n3[None, :], logits, -float("inf"))
        m_curr = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_curr - m_new)
        l_curr = tl.sum(tl.exp(logits - m_curr[:, None]), axis=1)
        l_i = l_i * alpha + l_curr * beta
        m_i = m_new
        is_target = offs_n3_curr[None, :] == target_y[:, None]
        target_logit += tl.sum(
            tl.where(is_target & mask_n3[None, :], logits, 0.0), axis=1
        )
        off_n3_start += BLOCK_N3
    lse = m_i + tl.log(l_i)
    tl.store(LSE + offs_n1, lse, mask=mask_n1)
    loss = (lse - target_logit).to(c_dtype)
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
    N1,
    N2: tl.constexpr,
    N3,
    HAS_BIAS: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_N3: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
):
    c_dtype = X.dtype.element_ty
    pid_n1 = tl.program_id(0)
    offs_n1 = pid_n1 * BLOCK_N1 + tl.arange(0, BLOCK_N1)
    mask_n1 = offs_n1 < N1
    lse = tl.load(LSE + offs_n1, mask=mask_n1, other=0.0)
    target_y = tl.load(Y + offs_n1, mask=mask_n1, other=0)
    grad_out = tl.load(GradOut + offs_n1, mask=mask_n1, other=0.0).to(p_dtype)
    x_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N1, N2),
        strides=(stride_x_n1, stride_x_n2),
        offsets=(pid_n1 * BLOCK_N1, 0),
        block_shape=(BLOCK_N1, BLOCK_N2),
        order=(1, 0),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        base=GradX,
        shape=(N1, N2),
        strides=(stride_x_n1, stride_x_n2),
        offsets=(pid_n1 * BLOCK_N1, 0),
        block_shape=(BLOCK_N1, BLOCK_N2),
        order=(1, 0),
    )
    offs_n3 = tl.arange(0, BLOCK_N3)
    off_n3_start = 0
    while off_n3_start < N3:
        offs_n3_curr = off_n3_start + offs_n3
        mask_n3 = offs_n3_curr < N3
        logits = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=p_dtype)
        if HAS_BIAS:
            bias_val = tl.load(B + offs_n3_curr * stride_b_n3, mask=mask_n3, other=0.0)
            logits += bias_val.to(p_dtype)[None, :]
        w_block_ptr = tl.make_block_ptr(
            base=W,
            shape=(N3, N2),
            strides=(stride_w_n3, stride_w_n2),
            offsets=(off_n3_start, 0),
            block_shape=(BLOCK_N3, BLOCK_N2),
            order=(1, 0),
        )
        x_ptr = x_block_ptr
        w_ptr = w_block_ptr
        for k in range(0, N2, BLOCK_N2):
            x_chunk = tl.load(x_ptr, boundary_check=(0, 1))
            w_chunk = tl.load(w_ptr, boundary_check=(0, 1))
            logits = tl.dot(
                x_chunk,
                tl.trans(w_chunk),
                logits,
                input_precision=INPUT_PRECISION,
                out_dtype=p_dtype,
            )
            x_ptr = tl.advance(x_ptr, (0, BLOCK_N2))
            w_ptr = tl.advance(w_ptr, (0, BLOCK_N2))
        logits = tl.where(mask_n3[None, :], logits, -float("inf"))
        probs = tl.exp(logits - lse[:, None])
        probs = tl.where(mask_n3[None, :], probs, 0.0)
        is_target = offs_n3_curr[None, :] == target_y[:, None]
        probs = tl.where(is_target, probs - 1.0, probs)
        grad_logits = (probs * grad_out[:, None]).to(c_dtype)
        g_ptr = grad_x_block_ptr
        w_ptr = w_block_ptr
        for k in range(0, N2, BLOCK_N2):
            w_chunk = tl.load(w_ptr, boundary_check=(0, 1))
            dx_part = tl.load(g_ptr, boundary_check=(0, 1))
            dx_part = tl.dot(
                grad_logits,
                w_chunk,
                dx_part,
                input_precision=INPUT_PRECISION,
                out_dtype=c_dtype,
            )
            tl.store(g_ptr, dx_part, boundary_check=(0, 1))
            w_ptr = tl.advance(w_ptr, (0, BLOCK_N2))
            g_ptr = tl.advance(g_ptr, (0, BLOCK_N2))
        off_n3_start += BLOCK_N3


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
    N1,
    N2: tl.constexpr,
    N3,
    HAS_BIAS: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_N3: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
):
    c_dtype = X.dtype.element_ty
    pid_n3 = tl.program_id(0)
    offs_n3 = pid_n3 * BLOCK_N3 + tl.arange(0, BLOCK_N3)
    mask_n3 = offs_n3 < N3
    offs_n1 = tl.arange(0, BLOCK_N1)
    off_n1_start = 0
    grad_w_block_ptr = tl.make_block_ptr(
        base=GradW,
        shape=(N3, N2),
        strides=(stride_w_n3, stride_w_n2),
        offsets=(pid_n3 * BLOCK_N3, 0),
        block_shape=(BLOCK_N3, BLOCK_N2),
        order=(1, 0),
    )
    w_block_ptr = tl.make_block_ptr(
        base=W,
        shape=(N3, N2),
        strides=(stride_w_n3, stride_w_n2),
        offsets=(pid_n3 * BLOCK_N3, 0),
        block_shape=(BLOCK_N3, BLOCK_N2),
        order=(1, 0),
    )
    while off_n1_start < N1:
        offs_n1_curr = off_n1_start + offs_n1
        mask_n1 = offs_n1_curr < N1
        lse = tl.load(LSE + offs_n1_curr, mask=mask_n1, other=0.0)
        target_y = tl.load(Y + offs_n1_curr, mask=mask_n1, other=0)
        grad_out = tl.load(GradOut + offs_n1_curr, mask=mask_n1, other=0.0).to(p_dtype)
        logits = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=p_dtype)
        if HAS_BIAS:
            bias_val = tl.load(B + offs_n3 * stride_b_n3, mask=mask_n3, other=0.0)
            logits += bias_val.to(p_dtype)[None, :]
        x_block_ptr = tl.make_block_ptr(
            base=X,
            shape=(N1, N2),
            strides=(stride_x_n1, stride_x_n2),
            offsets=(off_n1_start, 0),
            block_shape=(BLOCK_N1, BLOCK_N2),
            order=(1, 0),
        )
        x_ptr = x_block_ptr
        w_ptr = w_block_ptr
        for k in range(0, N2, BLOCK_N2):
            x_chunk = tl.load(x_ptr, boundary_check=(0, 1))
            w_chunk = tl.load(w_ptr, boundary_check=(0, 1))
            logits = tl.dot(
                x_chunk,
                tl.trans(w_chunk),
                logits,
                input_precision=INPUT_PRECISION,
                out_dtype=p_dtype,
            )
            x_ptr = tl.advance(x_ptr, (0, BLOCK_N2))
            w_ptr = tl.advance(w_ptr, (0, BLOCK_N2))
        logits = tl.where(mask_n1[:, None], logits, -float("inf"))
        probs = tl.exp(logits - lse[:, None])
        probs = tl.where(mask_n1[:, None], probs, 0.0)
        is_target = offs_n3[None, :] == target_y[:, None]
        probs = tl.where(is_target & mask_n1[:, None], probs - 1.0, probs)
        grad_logits_t = tl.trans((probs * grad_out[:, None]).to(c_dtype))
        x_ptr = x_block_ptr
        g_ptr = grad_w_block_ptr
        for k in range(0, N2, BLOCK_N2):
            x_chunk = tl.load(x_ptr, boundary_check=(0, 1))
            dw_part = tl.load(g_ptr, boundary_check=(0, 1))
            dw_part = tl.dot(
                grad_logits_t,
                x_chunk,
                dw_part,
                input_precision=INPUT_PRECISION,
                out_dtype=c_dtype,
            )
            tl.store(g_ptr, dw_part, boundary_check=(0, 1))
            x_ptr = tl.advance(x_ptr, (0, BLOCK_N2))
            g_ptr = tl.advance(g_ptr, (0, BLOCK_N2))
        off_n1_start += BLOCK_N1


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
    N1,
    N2: tl.constexpr,
    N3,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_N3: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
    HAS_BIAS: tl.constexpr = None,
):
    c_dtype = X.dtype.element_ty
    pid_n3 = tl.program_id(0)
    offs_n3 = pid_n3 * BLOCK_N3 + tl.arange(0, BLOCK_N3)
    mask_n3 = offs_n3 < N3
    db_acc = tl.zeros([BLOCK_N3], dtype=p_dtype)
    w_block_ptr = tl.make_block_ptr(
        base=W,
        shape=(N3, N2),
        strides=(stride_w_n3, stride_w_n2),
        offsets=(pid_n3 * BLOCK_N3, 0),
        block_shape=(BLOCK_N3, BLOCK_N2),
        order=(1, 0),
    )
    offs_n1 = tl.arange(0, BLOCK_N1)
    off_n1_start = 0
    while off_n1_start < N1:
        offs_n1_curr = off_n1_start + offs_n1
        mask_n1 = offs_n1_curr < N1
        target_y = tl.load(Y + offs_n1_curr, mask=mask_n1, other=0)
        lse = tl.load(LSE + offs_n1_curr, mask=mask_n1, other=0.0)
        grad_out = tl.load(GradOut + offs_n1_curr, mask=mask_n1, other=0.0).to(p_dtype)
        logits = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=p_dtype)
        bias_val = tl.load(B + offs_n3 * stride_b_n3, mask=mask_n3, other=0.0)
        logits += bias_val.to(p_dtype)[None, :]
        x_block_ptr = tl.make_block_ptr(
            base=X,
            shape=(N1, N2),
            strides=(stride_x_n1, stride_x_n2),
            offsets=(off_n1_start, 0),
            block_shape=(BLOCK_N1, BLOCK_N2),
            order=(1, 0),
        )
        x_ptr = x_block_ptr
        w_ptr = w_block_ptr
        for k in range(0, N2, BLOCK_N2):
            x_chunk = tl.load(x_ptr, boundary_check=(0, 1))
            w_chunk = tl.load(w_ptr, boundary_check=(0, 1))
            logits = tl.dot(
                x_chunk,
                tl.trans(w_chunk),
                logits,
                input_precision=INPUT_PRECISION,
                out_dtype=p_dtype,
            )
            x_ptr = tl.advance(x_ptr, (0, BLOCK_N2))
            w_ptr = tl.advance(w_ptr, (0, BLOCK_N2))
        logits = tl.where(mask_n1[:, None], logits, -float("inf"))
        probs = tl.exp(logits - lse[:, None])
        probs = tl.where(mask_n1[:, None], probs, 0.0)
        is_target = offs_n3[None, :] == target_y[:, None]
        probs = tl.where(is_target & mask_n1[:, None], probs - 1.0, probs)
        db_acc += tl.sum(probs * grad_out[:, None], 0)
        off_n1_start += BLOCK_N1
    tl.store(GradB + offs_n3 * stride_b_n3, db_acc.to(c_dtype), mask=mask_n3)


class TritonFusedNLL(torch.autograd.Function):
    TRITON_KWARGS = dict(BLOCK_N1=64, BLOCK_N2=64, BLOCK_N3=64, num_warps=4)

    @staticmethod
    def forward(ctx, x, target, weight, bias=None):
        assert x.ndim == 2, f"x must be 2D, got {x.ndim}"
        assert weight.ndim == 2, f"weight must be 2D, got {weight.ndim}"
        assert target.ndim == 1, f"target must be 1D, got {target.ndim}"
        assert x.shape[1] == weight.shape[1], (
            "x and weight must have the same hidden dimension"
        )
        assert x.shape[0] == target.shape[0], (
            "x and target must have the same batch size"
        )
        assert x.dtype == weight.dtype, "x and weight must have the same dtype"
        if bias is not None:
            assert bias.ndim == 1, f"bias must be 1D, got {bias.ndim}"
            assert bias.shape[0] == weight.shape[0], (
                "bias size must match weight output size"
            )
            assert bias.dtype == x.dtype, "bias must have the same dtype as x"
        target_i32 = target.to(torch.int32).contiguous()
        has_bias = bias is not None
        if not has_bias:
            bias = x.new_empty((1,), dtype=x.dtype)
        N1, N2, N3 = x.shape[0], x.shape[1], weight.shape[0]
        out = x.new_empty((N1,), dtype=x.dtype)
        lse = x.new_empty((N1,), dtype=TORCH_P_DTYPE)
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
