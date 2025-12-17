import time
from typing import Optional, Callable
import torch
from torch.utils.cpp_extension import load
# 加载编译好的 CUDA 扩展
lib = load(
    name="gemm_lib",
    sources=["gemm.cu"],
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
    A: torch.Tensor,
    B: torch.Tensor,
    tag: str,
    C: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
):
    if tag == "cpu":
        C = torch.zeros(A.shape[0], B.shape[1])
    else:
        C = torch.zeros(A.shape[0], B.shape[1], device='cuda')
    # Warm-up
    for _ in range(warmup):
        perf_func(A, B, C, A.shape[0], B.shape[1], A.shape[1])
    torch.cuda.synchronize()

    # Performance test
    start = time.time()
    for _ in range(iters):
        perf_func(A, B, C, A.shape[0], B.shape[1], A.shape[1])
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters

    # Output results
    print(f"Results for {tag}:")
    print(f"  Mean time per iteration: {mean_time:.2f} ms")

def test_gemm_functions(M: int, N: int, K: int):
    print("-" * 85)
    print(f"Testing GEMM functions with M={M}, N={N}, K={K}")
    print("-" * 85)

    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')

    print("Testing CPU GEMM...")
    A_cpu = A.cpu()
    B_cpu = B.cpu()
    C_cpu = torch.mm(A_cpu, B_cpu)
    run_benchmark(lambda A, B, C, M, N, K: lib.gemm(A, B, C, M, N, K), A_cpu, B_cpu, "cpu", warmup=10, iters=1000)


    print("Testing GEMM v1...")
    run_benchmark(lib.gemm_v1, A, B, "gemm_v1", warmup=10, iters=1000)
    print("Testing GEMM v2...")
    run_benchmark(lib.gemm_v2, A, B, "gemm_v2", warmup=10, iters=1000)
    print("Testing GEMM v3...")
    run_benchmark(lib.gemm_v3, A, B, "gemm_v3", warmup=10, iters=1000)

    C_gpu_v1 = torch.zeros_like(C_cpu, device='cuda')
    C_gpu_v2 = torch.zeros_like(C_cpu, device='cuda')
    C_gpu_v3 = torch.zeros_like(C_cpu, device='cuda')
    lib.gemm_v1(A, B, C_gpu_v1, M, N, K)
    lib.gemm_v2(A, B, C_gpu_v2, M, N, K)
    lib.gemm_v3(A, B, C_gpu_v3, M, N, K)
    #assert torch.allclose(C_cpu, C_gpu_v1.cpu(), atol=1e-5), "gemm_v1 result is incorrect"
    #assert torch.allclose(C_cpu, C_gpu_v2.cpu(), atol=1e-5), "gemm_v2 result is incorrect"
    #assert torch.allclose(C_cpu, C_gpu_v3.cpu(), atol=1e-5), "gemm_v3 result is incorrect"

if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    test_gemm_functions(M, N, K)