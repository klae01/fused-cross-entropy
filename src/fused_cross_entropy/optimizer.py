import math
from itertools import product

import torch
import triton

from .fused_nll import (
    TORCH_P_DTYPE,
    _fused_nll_bwd_dw_db_kernel,
    _fused_nll_bwd_dx_kernel,
    _fused_nll_fwd_kernel,
)


def round_up(x, y):
    return math.ceil(x / y) * y


def priority_fn1(N1, N3, SM, BLOCK_N1, BLOCK_N3, num_warps, num_stages, **kwargs):
    c1 = math.ceil(math.ceil(N3 / BLOCK_N3)) * round_up(N1, BLOCK_N1)
    c2 = round_up(math.ceil(N1 / BLOCK_N1), SM) * round_up(N3, BLOCK_N3)
    return (c1 + c2, num_stages <= 1, -num_warps, -num_stages)


def priority_fn2(N1, N3, SM, BLOCK_N1, BLOCK_N3, num_warps, num_stages, **kwargs):
    c1 = math.ceil(math.ceil(N3 / BLOCK_N3)) * round_up(N1, BLOCK_N1) * 3
    c2 = round_up(math.ceil(N1 / BLOCK_N1), SM) * round_up(N3, BLOCK_N3) * 2
    return (c1 + c2, num_stages <= 1, -num_warps, -num_stages)


def priority_fn3(N1, N3, SM, BLOCK_N1, BLOCK_N3, num_warps, num_stages, **kwargs):
    c1 = round_up(math.ceil(N3 / BLOCK_N3), SM) * round_up(N1, BLOCK_N1) * 2
    c2 = math.ceil(math.ceil(N1 / BLOCK_N1)) * round_up(N3, BLOCK_N3) * 3
    return (c1 + c2, num_stages <= 1, -num_warps, -num_stages)


class KernelOptimizer:
    _CONFIG_CACHE = {}

    @classmethod
    def get_best_config(cls, kernel_fn, args, kwargs, cost_fn):
        N1 = kwargs["N1"]
        N2 = kwargs["N2"]
        N3 = kwargs["N3"]
        kwargs = dict(kwargs)
        element_size = args[0].element_size()
        device = args[0].device
        device_props = torch.cuda.get_device_properties(device)
        SM = device_props.multi_processor_count
        max_smem = device_props.shared_memory_per_block_optin
        arg_dtypes = tuple(getattr(a, "dtype", type(a)) for a in args)
        kwarg_items = tuple(sorted(kwargs.items()))
        cache_key = (kernel_fn.__name__, arg_dtypes, kwarg_items, device)
        if cache_key in cls._CONFIG_CACHE:
            return cls._CONFIG_CACHE[cache_key]
        candidates_n1 = [32, 64, 128, 256]
        candidates_n3 = [32, 64, 128, 256]
        candidates_warps = [8]
        candidates_stages = [1, 2, 3, 4, 5]
        fixed_n2 = max(32, int(128 // element_size))
        configs = []
        for n1, n3, warps, stages in product(
            candidates_n1, candidates_n3, candidates_warps, candidates_stages
        ):
            cfg = dict(
                BLOCK_N1=n1,
                BLOCK_N2=fixed_n2,
                BLOCK_N3=n3,
                num_warps=warps,
                num_stages=stages,
            )
            cfg["priority"] = cost_fn(N1=N1, N2=N2, N3=N3, SM=SM, **cfg)
            configs.append(cfg)
        configs.sort(key=lambda x: x["priority"])
        for cfg in configs:
            cfg_args = {k: v for k, v in cfg.items() if k != "priority"}
            kernel = kernel_fn.warmup(*args, **kwargs, **cfg_args, grid=(1,))
            if kernel.metadata.shared > max_smem:
                continue
            kernel._init_handles()
            if kernel.n_spills == 0:
                cls._CONFIG_CACHE[cache_key] = cfg_args
                return cfg_args
        fallback = {
            "BLOCK_N1": 32,
            "BLOCK_N2": 32,
            "BLOCK_N3": 32,
            "num_warps": 8,
            "num_stages": 2,
        }
        cls._CONFIG_CACHE[cache_key] = fallback
        return fallback


class TritonFusedNLL(torch.autograd.Function):
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
        N1 = x.shape[0]
        N2 = x.shape[1]
        N3 = weight.shape[0]
        out = torch.empty((N1,), device=x.device, dtype=x.dtype)
        lse = torch.empty((N1,), device=x.device, dtype=TORCH_P_DTYPE)
        common_kwargs = {
            "stride_x_n1": x.stride(0),
            "stride_x_n2": x.stride(1),
            "stride_w_n3": weight.stride(0),
            "stride_w_n2": weight.stride(1),
            "stride_b_n3": bias.stride(0),
            "N1": N1,
            "N2": N2,
            "N3": N3,
            "HAS_BIAS": has_bias,
        }
        fwd_args = (x, target_i32, weight, bias, out, lse)
        config = KernelOptimizer.get_best_config(
            _fused_nll_fwd_kernel, fwd_args, common_kwargs, priority_fn1
        )
        _fused_nll_fwd_kernel[lambda meta: (triton.cdiv(N1, meta["BLOCK_N1"]),)](
            *fwd_args, **common_kwargs, **config
        )
        ctx.save_for_backward(x, target_i32, weight, bias, lse)
        ctx.has_bias = has_bias
        ctx.common_kwargs = common_kwargs
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, target_i32, weight, bias, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        N1 = x.shape[0]
        N3 = weight.shape[0]
        grad_x = torch.zeros_like(x)
        grad_w = torch.zeros_like(weight)
        grad_b = torch.zeros_like(bias) if ctx.has_bias else None
        common_kwargs = ctx.common_kwargs
        dx_args = (x, target_i32, grad_output, weight, bias, grad_x, lse)
        config_dx = KernelOptimizer.get_best_config(
            _fused_nll_bwd_dx_kernel, dx_args, common_kwargs, priority_fn2
        )
        _fused_nll_bwd_dx_kernel[lambda meta: (triton.cdiv(N1, meta["BLOCK_N1"]),)](
            *dx_args, **common_kwargs, **config_dx
        )
        dw_db_args = (x, target_i32, grad_output, weight, bias, grad_w, grad_b, lse)
        config_dw_db = KernelOptimizer.get_best_config(
            _fused_nll_bwd_dw_db_kernel, dw_db_args, common_kwargs, priority_fn3
        )
        _fused_nll_bwd_dw_db_kernel[lambda meta: (triton.cdiv(N3, meta["BLOCK_N3"]),)](
            *dw_db_args, **common_kwargs, **config_dw_db
        )
        return grad_x, None, grad_w, grad_b


def triton_fused_nll(x, target, weight, bias=None):
    return TritonFusedNLL.apply(x, target, weight, bias)
