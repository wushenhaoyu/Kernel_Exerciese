import time
import torch
from torch.utils.cpp_extension import load

# 加载自定义的 CUDA 扩展
lib = load(
    name="embedding_lib",
    sources=["embedding.cu"],  # 确保文件名与你的 CUDA 文件一致
    extra_cuda_cflags=[
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

torch.set_grad_enabled(False)

def run_benchmark(perf_func, input_tensor, wte_tensor, wpe_tensor, out_tensor, tag, warmup=10, iters=1000):
    device = input_tensor.device
    start = time.time()
    for _ in range(warmup):
        perf_func(out_tensor, input_tensor, wte_tensor, wpe_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        perf_func(out_tensor, input_tensor, wte_tensor, wpe_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    print(f"Results for {tag}:")
    print(f"  Mean time per iteration: {mean_time:.2f} ms")
    print(f"  Output: {out_tensor.cpu().flatten()[:10]}...")

def test_embedding_functions(batch_sizes, seq_lengths, embedding_dims):
    print("-" * 85)
    print("Testing Embedding functions")
    print("-" * 85)

    vocab_size = 100  # 假设词汇表大小为100
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            for embedding_dim in embedding_dims:
                print(f"Testing with batch size: {batch_size}, sequence length: {seq_length}, embedding dim: {embedding_dim}")
                # 测试 CPU 版本
                input_cpu = torch.randint(0, vocab_size, (batch_size, seq_length), device='cpu', dtype=torch.int)
                wte_cpu = torch.randn(vocab_size, embedding_dim, device='cpu')
                wpe_cpu = torch.randn(seq_length, embedding_dim, device='cpu')
                out_cpu = torch.zeros((batch_size, seq_length, embedding_dim), device='cpu')
                run_benchmark(lib.encoder, input_cpu, wte_cpu, wpe_cpu, out_cpu, "Encoder_cpu", warmup=0, iters=1)

                # 测试 GPU 版本
                input_cuda = input_cpu.to('cuda')
                wte_cuda = wte_cpu.to('cuda')
                wpe_cuda = wpe_cpu.to('cuda')
                out_cuda = out_cpu.to('cuda')

                # Test Encoder_v1
                run_benchmark(lib.encoder_v1, input_cuda, wte_cuda, wpe_cuda, out_cuda, "Encoder_v1",  warmup=10, iters=1000)

                # Test Encoder_v2
                run_benchmark(lib.encoder_v2, input_cuda, wte_cuda, wpe_cuda, out_cuda, "Encoder_v2",  warmup=10, iters=1000)
if __name__ == "__main__":
    batch_sizes = [128, 256]
    seq_lengths = [64, 128]
    embedding_dims = [512, 1024]
    test_embedding_functions(batch_sizes, seq_lengths, embedding_dims)