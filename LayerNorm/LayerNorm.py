import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load

# 加载自定义的 CUDA 扩展
lib = load(
    name="layernorm_lib",
    sources=["LayerNorm.cu"],  # 确保文件名与你的 CUDA 文件一致
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
    weight: torch.Tensor,
    bias: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    mean: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
):
    device = x.device
    out = torch.zeros_like(x)
    mean = torch.zeros(x.size(0), x.size(1), device=device)
    rstd = torch.zeros(x.size(0), x.size(1), device=device)

    # Warm-up
    for _ in range(warmup):
        perf_func(out, mean, rstd, x, weight, bias)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Performance test
    start = time.time()
    for _ in range(iters):
        perf_func(out, mean, rstd, x, weight, bias)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters

    # Output results
    print(f"Results for {tag}:")
    print(f"  Mean time per iteration: {mean_time:.2f} ms")
    print(f"  Output: {out.flatten()[:10]}...")  


def test_layernorm_functions(input_sizes: list):
    print("-" * 85)
    print("Testing LayerNorm functions")
    print("-" * 85)

    for input_size in input_sizes:
        print(f"Testing with input size: {input_size}")
        # 测试 CPU 版本
        x_cpu = torch.randn(input_size, device='cpu')
        weight_cpu = torch.randn(input_size[-1], device='cpu')
        bias_cpu = torch.randn(input_size[-1], device='cpu')
        run_benchmark(lib.LayerNorm, x_cpu, weight_cpu, bias_cpu, "LayerNorm_cpu", warmup=0, iters=1)

        # 测试 GPU 版本
        x_cuda = torch.randn(input_size, device='cuda')
        weight_cuda = torch.randn(input_size[-1], device='cuda')
        bias_cuda = torch.randn(input_size[-1], device='cuda')

        # Test LayerNorm_v1
        run_benchmark(lib.LayerNorm_v1, x_cuda, weight_cuda, bias_cuda, "LayerNorm_v1", warmup=10, iters=1000)

        # Test LayerNorm_v2
        run_benchmark(lib.LayerNorm_v2, x_cuda, weight_cuda, bias_cuda, "LayerNorm_v2", warmup=10, iters=1000)

        # Test LayerNorm_v3
        run_benchmark(lib.LayerNorm_v3, x_cuda, weight_cuda, bias_cuda, "LayerNorm_v3", warmup=10, iters=1000)

        # Test LayerNorm_v4
        run_benchmark(lib.LayerNorm_v4, x_cuda, weight_cuda, bias_cuda, "LayerNorm_v4", warmup=10, iters=1000)

        # Test LayerNorm_v5
        run_benchmark(lib.LayerNorm_v5, x_cuda, weight_cuda, bias_cuda, "LayerNorm_v5", warmup=10, iters=1000)


if __name__ == "__main__":
    input_sizes = [(128, 64, 512), (128, 64, 1024), (128, 64, 2048)]  # 示例输入尺寸
    test_layernorm_functions(input_sizes)