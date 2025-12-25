import time
from typing import Optional, Callable
import torch
from torch.utils.cpp_extension import load
lib = load(
    name="softmax_lib",
    sources=["softmax.cu"],
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


def run_benchmark(
    perf_func: Callable,
    input: torch.Tensor,
    tag: str,
    output: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
):
    if tag == "cpu":
        output = torch.zeros_like(input)
    else:
        output = torch.zeros_like(input, device='cuda')
    for _ in range(warmup):
        perf_func(input, output)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        perf_func(input, output)    
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000
    mean_time = total_time / iters
    print(f"Results for {tag}:")
    print(f"  Mean time per iteration: {mean_time:.2f} ms")


def test_softmax_functions(M: int, N: int) -> None:
    print("-" * 85)
    print(f"Testing Softmax functions with M={M}, N={N}")
    print("-" * 85)
    input = torch.randn(M, N, device='cuda')
    input = input.cpu()
    print("Testing CPU Softmax...")
    run_benchmark(lib.softmax, input, tag="cpu", warmup=0, iters=1)
    input = input.cuda()
    print("Testing softmax_v1...")
    run_benchmark(lib.softmax_v1, input, tag="softmax_v1")
    print("Testing softmax_v2...")
    run_benchmark(lib.softmax_v2, input, tag="softmax_v2")
    input = input.cpu()
    output_cpu = torch.zeros_like(input)
    lib.softmax(input, output_cpu)
    input = input.cuda()
    output_gpu_v1 = torch.zeros_like(input)
    output_gpu_v2 = torch.zeros_like(input)
    lib.softmax_v1(input, output_gpu_v1)
    lib.softmax_v2(input, output_gpu_v2)
    output_gpu_v1_cpu = output_gpu_v1.cpu()
    output_gpu_v2_cpu = output_gpu_v2.cpu()
    #print("Verifying results...")
    #print("cpu:", output_cpu)
    #print("gpu_v1:", output_gpu_v1_cpu)
    #print("gpu_v2:", output_gpu_v2_cpu)
    assert torch.allclose(output_cpu, output_gpu_v1_cpu, atol=1e-3), "softmax_v1 result is incorrect"
    assert torch.allclose(output_cpu, output_gpu_v2_cpu, atol=1e-3), "softmax_v2 result is incorrect"

if __name__ == "__main__":
    test = [(1024, 1024), (2048, 2048), (4096, 4096)]
    for M, N in test:
        test_softmax_functions(M, N)
