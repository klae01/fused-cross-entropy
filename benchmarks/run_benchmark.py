import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import version
from typing import Dict

import liger_kernel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

from fused_cross_entropy import triton_fused_nll


def get_versions() -> Dict[str, str]:
    versions = {}
    versions["PyTorch"] = torch.__version__

    try:
        versions["Ours"] = version("fused-cross-entropy")
    except Exception:
        versions["Ours"] = "N/A"

    try:
        versions["Liger-Kernel"] = version("liger-kernel")
    except Exception:
        versions["Liger-Kernel"] = "N/A"

    return versions


def get_candidate_label(candidate: str, versions: Dict[str, str]) -> str:
    if candidate == "Ours":
        return f"Ours ({versions['Ours']})"
    elif candidate == "Liger-Kernel":
        return f"Liger-Kernel ({versions['Liger-Kernel']})"
    elif candidate == "PyTorch":
        return f"PyTorch ({versions['PyTorch']})"
    elif candidate == "Torch Compiled":
        return f"Torch Compiled ({versions['PyTorch']})"
    return candidate


def run_iteration(candidate: str, x, target, weight, bias):
    if candidate == "Ours":
        loss = triton_fused_nll(x, target, weight, bias)
        loss.sum().backward()

    elif candidate == "Liger-Kernel":
        # Liger signature: input, weight, target, bias
        loss = liger_fused_linear_cross_entropy(x, weight, target, bias=bias)
        loss.backward()

    elif candidate == "PyTorch":
        logits = F.linear(x, weight, bias)
        loss = F.cross_entropy(logits, target)
        loss.backward()

    elif candidate == "Torch Compiled":
        if not hasattr(run_iteration, "_compiled_fn"):

            def fn(x, t, w, b):
                logits = F.linear(x, w, b)
                return F.cross_entropy(logits, t)

            run_iteration._compiled_fn = torch.compile(fn)

        loss = run_iteration._compiled_fn(x, target, weight, bias)
        loss.backward()
    else:
        raise ValueError(f"Unknown candidate: {candidate}")


def run_worker_mode(args):
    """
    Worker process to run a single benchmark point.
    Outputs JSON to stdout.
    """
    try:
        device = torch.device("cuda")
        if args.dtype == "float16":
            dtype = torch.float16
        elif args.dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        B, H, V = args.B, args.H, args.V
        candidate = args.candidate

        # Seed for reproducibility
        torch.manual_seed(42)

        # Initialize Tensors
        # We wrap in try-except to catch OOM during initialization
        try:
            x = torch.randn(B, H, device=device, dtype=dtype, requires_grad=True)
            weight = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
            bias = torch.randn(V, device=device, dtype=dtype, requires_grad=True)
            target = torch.randint(0, V, (B,), device=device)
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"error": "OOM_INIT"}))
            return

        # Calculate Static Memory (X + W + b)
        # Note: We exclude target from static memory calculation as per prompt instructions,
        # though it technically takes memory.
        static_mem_bytes = (
            x.element_size() * x.numel() * 2
            + weight.element_size() * weight.numel() * 2
            + bias.element_size() * bias.numel() * 2
        )

        # Warmup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(5):
                run_iteration(candidate, x, target, weight, bias)
                # Zero grads
                x.grad = None
                weight.grad = None
                bias.grad = None
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"error": "OOM_WARMUP"}))
            return
        except Exception as e:
            print(json.dumps({"error": f"RUNTIME_ERROR: {str(e)}"}))
            return

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        try:
            start_event.record()
            run_iteration(candidate, x, target, weight, bias)
            end_event.record()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"error": "OOM_EXEC"}))
            return

        latency_ms = start_event.elapsed_time(end_event)
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        overhead_bytes = peak_mem_bytes - static_mem_bytes

        result = {
            "latency_ms": latency_ms,
            "memory_overhead_mb": overhead_bytes / (1024 * 1024),
            "memory_overhead_gb": overhead_bytes / (1024 * 1024 * 1024),
            "peak_memory_gb": peak_mem_bytes / (1024 * 1024 * 1024),
            "error": None,
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": f"UNHANDLED: {str(e)}"}))


def run_orchestrator_mode(args):
    """
    Orchestrator to run all experiments and plot results.
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    versions = get_versions()
    candidates = ["Ours", "PyTorch", "Torch Compiled"]
    if liger_kernel is not None:
        candidates.insert(1, "Liger-Kernel")

    # Default Config
    # B = 2^13 = 8192
    # H = 2^12 = 4096
    # V = 2^16 = 65536
    base_B = 13
    base_H = 12
    base_V = 16

    experiments = [
        {
            "name": "Experiment A: Varying Batch Size (B)",
            "vary": "B",
            "range": range(11, 16),  # 2^11 (2k) to 2^15 (32k)
            "fixed": {"H": 2**base_H, "V": 2**base_V},
            "x_label": "Batch Size (B)",
        },
        {
            "name": "Experiment B: Varying Hidden Dim (H)",
            "vary": "H",
            "range": range(9, 14),  # 2^9 (512) to 2^13 (8k)
            "fixed": {"B": 2**base_B, "V": 2**base_V},
            "x_label": "Hidden Dim (H)",
        },
        {
            "name": "Experiment C: Varying Vocab Size (V)",
            "vary": "V",
            "range": range(14, 19),  # 2^14 (16k) to 2^18 (262k)
            "fixed": {"B": 2**base_B, "H": 2**base_H},
            "x_label": "Vocab Size (V)",
        },
    ]

    results = []

    for exp in experiments:
        print(f"\nRunning {exp['name']}...")
        vary_key = exp["vary"]
        fixed = exp["fixed"]

        for p in exp["range"]:
            val = 2**p
            current_config = fixed.copy()
            current_config[vary_key] = val

            B, H, V = current_config["B"], current_config["H"], current_config["V"]

            for cand in candidates:
                print(f"  Running {cand} | B={B}, H={H}, V={V} ... ", end="")
                sys.stdout.flush()

                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--mode",
                    "worker",
                    "--candidate",
                    cand,
                    "--B",
                    str(B),
                    "--H",
                    str(H),
                    "--V",
                    str(V),
                    "--dtype",
                    args.dtype,
                ]

                try:
                    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                    data = json.loads(out.decode("utf-8").strip().split("\n")[-1])

                    if data.get("error"):
                        print(f"FAILED ({data['error']})")
                        continue

                    print(
                        f"Done ({data['latency_ms']:.2f}ms, {data['memory_overhead_mb']:.2f}MB)"
                    )

                    results.append(
                        {
                            "Experiment": exp["name"],
                            "Candidate": get_candidate_label(cand, versions),
                            "X_Value": val,
                            "Latency (ms)": data["latency_ms"],
                            "Memory Overhead (GB)": data["memory_overhead_gb"],
                        }
                    )

                except subprocess.CalledProcessError as e:
                    print(f"CRASHED: {e.output.decode('utf-8')}")
                except Exception as e:
                    print(f"ERROR: {e}")

    # Plotting
    df = pd.DataFrame(results)
    if df.empty:
        print("No results collected.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Latency, Row 2: Memory
    metrics = ["Latency (ms)", "Memory Overhead (GB)"]

    for i, metric in enumerate(metrics):
        for j, exp in enumerate(experiments):
            ax = axes[i, j]
            exp_name = exp["name"]
            subset = df[df["Experiment"] == exp_name]

            if subset.empty:
                continue

            sns.lineplot(
                data=subset,
                x="X_Value",
                y=metric,
                hue="Candidate",
                style="Candidate",
                markers=True,
                dashes=False,
                ax=ax,
            )

            ax.set_xscale("log", base=2)
            ax.set_xlabel(exp["x_label"])
            fixed_str = " ".join([f"{k}={v}" for k, v in exp["fixed"].items()])
            ax.set_title(f"{metric} vs {exp['vary']}\n({fixed_str} dtype={args.dtype})")
            if j > 0:  # Only show legend on first column to save space
                ax.get_legend().remove()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "benchmark_results.png")
    plt.savefig(plot_path)
    print(f"\nPlots saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["worker", "orchestrator"], default="orchestrator"
    )
    parser.add_argument("--candidate", type=str)
    parser.add_argument("--B", type=int)
    parser.add_argument("--H", type=int)
    parser.add_argument("--V", type=int)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
    )
    parser.add_argument("--output_dir", type=str, default="benchmark_results")

    args = parser.parse_args()

    if args.mode == "worker":
        run_worker_mode(args)
    else:
        run_orchestrator_mode(args)
