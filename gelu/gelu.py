import time
import torch
from torch.utils.cpp_extension import load

# 加载 CUDA 扩展
lib = load(
    name="gelu_lib",
    sources=["gelu.cu"],
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

def run_benchmark(perf_func, input_tensor, tag, warmup=10, iters=1000):
    """
    运行基准测试，测量 GELU 操作的性能。
    """
    # Warm-up
    for _ in range(warmup):
        perf_func(input_tensor)
    torch.cuda.synchronize()

    # 性能测试
    start = time.time()
    for _ in range(iters):
        perf_func(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # 转换为毫秒
    mean_time = total_time / iters

    # 输出结果
    print(f"Results for {tag}:")
    print(f"  Mean time per iteration: {mean_time:.2f} ms")


def test_gelu_functions(input_size):
    """
    测试 GELU 函数的性能和正确性。
    """
    print("-" * 85)
    print(f"Testing GELU functions with input size: {input_size}")
    print("-" * 85)

    # 测试 CUDA 上的 GELU v1 (float32)
    print("Testing GELU v1 (float32) on CUDA...")
    input_tensor_v1 = torch.randn(input_size, device="cuda", dtype=torch.float32)
    run_benchmark(lambda x: lib.gelu_v1(x, torch.empty_like(x)), input_tensor_v1, "gelu_v1 (float32)", warmup=10, iters=1000)

    # 测试 CUDA 上的 GELU v2 (float16)
    print("Testing GELU v2 (float16) on CUDA...")
    input_tensor_v2 = torch.randn(input_size, device="cuda", dtype=torch.float16)
    run_benchmark(lambda x: lib.gelu_v2(x, torch.empty_like(x)), input_tensor_v2, "gelu_v2 (float16)", warmup=10, iters=1000)

    # 测试 CPU 上的 GELU 函数 (float32)
    print("Testing CPU GELU function (float32)...")
    input_tensor_cpu = torch.randn(input_size, device="cpu", dtype=torch.float32)
    output_tensor_cpu = torch.empty_like(input_tensor_cpu)
    lib.gelu(input_tensor_cpu, output_tensor_cpu)

    # 验证 CPU GELU 函数的正确性
    cpu_output = torch.nn.functional.gelu(input_tensor_cpu)
    assert torch.allclose(cpu_output, output_tensor_cpu, atol=1e-3), "CPU GELU result is incorrect"
    print("CPU GELU function is correct.")


if __name__ == "__main__":
    input_size = 1024 * 1024  # 输入张量大小
    test_gelu_functions(input_size)