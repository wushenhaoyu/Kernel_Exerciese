import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load

lib = load(
    name="reduce_lib",
    sources=["reduce.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

torch.set_grad_enabled(False)

def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
):
    if tag == "cpu":
        out = torch.zeros(1)
    else:
        out = torch.zeros(1, device='cuda')
    # Warm-up
    for _ in range(warmup):
        perf_func(x, out, x.numel())
    torch.cuda.synchronize()

    # Performance test
    start = time.time()
    for _ in range(iters):
        perf_func(x, out, x.numel())
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters

    out.fill_(0)
    perf_func(x, out, x.numel())

    # Output results
    print(f"Results for {tag}:")
    print(f"  Mean time per iteration: {mean_time:.2f} ms")
    print(f"  Output: {out}")

def test_reduce_functions(input_sizes: list, S: int, K: int):
    print("-" * 85)
    print(f"Testing reduce functions with S={S}, K={K}")
    print("-" * 85)

    for input_size in input_sizes:
        print(f"Testing with input size: {input_size}")
        x = torch.randn(input_size, device='cuda')
        y = torch.zeros(1, device='cuda')

        run_benchmark(lib.reduce, x, "cpu", out=y, warmup=2, iters=10)
        # Test reduce_v1
        run_benchmark(lib.reduce_v1, x, "reduce_v1", out=y, warmup=10, iters=1000)

        # Test reduce_v2
        run_benchmark(lib.reduce_v2, x, "reduce_v2", out=y, warmup=10, iters=1000)

        # Test reduce_v3
        run_benchmark(lib.reduce_v3, x, "reduce_v3", out=y, warmup=10, iters=1000)

        # Test reduce_v4
        run_benchmark(lib.reduce_v4, x, "reduce_v4", out=y, warmup=10, iters=1000)

        # Test reduce_v5
        run_benchmark(lib.reduce_v5, x, "reduce_v5", out=y, warmup=10, iters=1000)

if __name__ == "__main__":
    input_sizes = [1024, 1024 * 1024, 1024 * 1024 * 10]
    S = 1024  # Example parameter
    K = 1024  # Example parameter
    test_reduce_functions(input_sizes, S, K)