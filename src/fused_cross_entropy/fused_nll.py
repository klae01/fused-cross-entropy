import torch
import triton
import triton.language as tl

TORCH_P_DTYPE = torch.float32
TRITON_P_DTYPE = tl.float32


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
    m_i = tl.zeros([BLOCK_N1], dtype=TRITON_P_DTYPE) - float("inf")
    l_i = tl.zeros([BLOCK_N1], dtype=TRITON_P_DTYPE)
    target_logit = tl.zeros([BLOCK_N1], dtype=TRITON_P_DTYPE)
    x_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N1, N2),
        strides=(stride_x_n1, stride_x_n2),
        offsets=(pid * BLOCK_N1, 0),
        block_shape=(BLOCK_N1, BLOCK_N2),
        order=(1, 0),
    )
    offs_n3_curr = tl.arange(0, BLOCK_N3)
    off_n3_start = 0
    while off_n3_start < N3:
        mask_n3 = offs_n3_curr < N3
        if HAS_BIAS:
            bias_val = tl.load(B + offs_n3_curr * stride_b_n3, mask=mask_n3, other=0.0)
            logits = tl.broadcast_to(
                bias_val.to(TRITON_P_DTYPE)[None, :], [BLOCK_N1, BLOCK_N3]
            )
        else:
            logits = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=TRITON_P_DTYPE)
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
                out_dtype=TRITON_P_DTYPE,
            )
            x_ptr = tl.advance(x_ptr, (0, BLOCK_N2))
            w_ptr = tl.advance(w_ptr, (0, BLOCK_N2))
        logits = tl.where(mask_n3[None, :], logits, -float("inf"))
        m_curr = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        l_curr = tl.sum(tl.exp(logits - m_new[:, None]), axis=1)
        l_i = l_i * alpha + l_curr
        m_i = m_new
        is_target = offs_n3_curr[None, :] == target_y[:, None]
        target_logit += tl.sum(
            tl.where(is_target & mask_n3[None, :], logits, 0.0), axis=1
        )
        off_n3_start += BLOCK_N3
        offs_n3_curr += BLOCK_N3
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
    grad_out = tl.load(GradOut + offs_n1, mask=mask_n1, other=0.0).to(TRITON_P_DTYPE)
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
    offs_n3_curr = tl.arange(0, BLOCK_N3)
    off_n3_start = 0
    while off_n3_start < N3:
        mask_n3 = offs_n3_curr < N3
        if HAS_BIAS:
            bias_val = tl.load(B + offs_n3_curr * stride_b_n3, mask=mask_n3, other=0.0)
            logits = tl.broadcast_to(
                bias_val.to(TRITON_P_DTYPE)[None, :], [BLOCK_N1, BLOCK_N3]
            )
        else:
            logits = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=TRITON_P_DTYPE)
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
                out_dtype=TRITON_P_DTYPE,
            )
            x_ptr = tl.advance(x_ptr, (0, BLOCK_N2))
            w_ptr = tl.advance(w_ptr, (0, BLOCK_N2))
        probs = tl.exp(logits - lse[:, None])
        probs -= offs_n3_curr[None, :] == target_y[:, None]
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
        offs_n3_curr += BLOCK_N3


@triton.jit
def _fused_nll_bwd_dw_db_kernel(
    X,
    Y,
    GradOut,
    W,
    B,
    GradW,
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
    offs_n1_curr = tl.arange(0, BLOCK_N1)
    off_n1_start = 0
    if HAS_BIAS:
        db_acc = tl.zeros([BLOCK_N3], dtype=TRITON_P_DTYPE)
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
        mask_n1 = offs_n1_curr < N1
        lse = tl.load(LSE + offs_n1_curr, mask=mask_n1, other=0.0)
        target_y = tl.load(Y + offs_n1_curr, mask=mask_n1, other=0)
        grad_out = tl.load(GradOut + offs_n1_curr, mask=mask_n1, other=0.0).to(
            TRITON_P_DTYPE
        )
        if HAS_BIAS:
            bias_val = tl.load(B + offs_n3 * stride_b_n3, mask=mask_n3, other=0.0)
            logits = tl.broadcast_to(
                bias_val.to(TRITON_P_DTYPE)[None, :], [BLOCK_N1, BLOCK_N3]
            )
        else:
            logits = tl.zeros([BLOCK_N1, BLOCK_N3], dtype=TRITON_P_DTYPE)
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
                out_dtype=TRITON_P_DTYPE,
            )
            x_ptr = tl.advance(x_ptr, (0, BLOCK_N2))
            w_ptr = tl.advance(w_ptr, (0, BLOCK_N2))
        probs = tl.exp(logits - lse[:, None])
        probs -= offs_n3[None, :] == target_y[:, None]
        d_logits = probs * grad_out[:, None]
        if HAS_BIAS:
            db_acc += tl.sum(d_logits, 0)
        d_logits_t = tl.trans(d_logits.to(c_dtype))
        x_ptr = x_block_ptr
        g_ptr = grad_w_block_ptr
        for k in range(0, N2, BLOCK_N2):
            x_chunk = tl.load(x_ptr, boundary_check=(0, 1))
            dw_part = tl.load(g_ptr, boundary_check=(0, 1))
            dw_part = tl.dot(
                d_logits_t,
                x_chunk,
                dw_part,
                input_precision=INPUT_PRECISION,
                out_dtype=c_dtype,
            )
            tl.store(g_ptr, dw_part, boundary_check=(0, 1))
            x_ptr = tl.advance(x_ptr, (0, BLOCK_N2))
            g_ptr = tl.advance(g_ptr, (0, BLOCK_N2))
        off_n1_start += BLOCK_N1
        offs_n1_curr += BLOCK_N1
    if HAS_BIAS:
        tl.store(GradB + offs_n3 * stride_b_n3, db_acc.to(c_dtype), mask=mask_n3)
