#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;



void __global__ reduce_kernel_v1(float *d_x, float *d_y)
{
    const int tid = threadIdx.x;
    float *x = d_x + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(d_y, x[0]);
    }
}


void __global__ reduce_kernel_v2(float *d_x, float *d_y, int N)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_y[];    
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;    
    __syncthreads();
    

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(d_y, s_y[0]);
    }
}


void __global__ reduce_kernel_v3(float *d_x, float *d_y, int N)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_y[];    
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;    
    __syncthreads();
    

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    for (int offset = 16 ; offset > 0 ; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncwarp();
    }

    if (tid == 0)
    {
        atomicAdd(d_y, s_y[0]);
    }
}



void __global__ reduce_kernel_v4(float *d_x, float *d_y, int N)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_y[];    
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;    
    __syncthreads();
    

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    float y = s_y[tid];

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        y += __shfl_down_sync(0xffffffff, y, offset);
    }


    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}


void __global__ reduce_kernel_v5(float *d_x, float *d_y, int N)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_y[];    
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;    
    __syncthreads();
    

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    float y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }


    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}


// 定义Python接口

void reduce(torch::Tensor x, torch::Tensor y, int N) {

    auto x_cpu = x.to(torch::kCPU);
    auto y_cpu = y.to(torch::kCPU);


    float* x_data = x_cpu.data_ptr<float>();
    float* y_data = y_cpu.data_ptr<float>();


    y_data[0] = 0.0;


    for (int i = 0; i < N; ++i) {
        y_data[0] += x_data[i];
    }

    y.copy_(y_cpu);
}

void reduce_v1(torch::Tensor d_x, torch::Tensor d_y, int N)
{
    const int blockSize = 256;
    const int numBlocks = (d_x.numel() + blockSize - 1) / blockSize;
    reduce_kernel_v1<<<numBlocks, blockSize>>>(d_x.data_ptr<float>(), d_y.data_ptr<float>());
    cudaDeviceSynchronize();
}

void reduce_v2(torch::Tensor d_x, torch::Tensor d_y, int N)
{
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    reduce_kernel_v2<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_x.data_ptr<float>(), d_y.data_ptr<float>(), N);
    cudaDeviceSynchronize();
}

void reduce_v3(torch::Tensor d_x, torch::Tensor d_y, int N)
{
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    reduce_kernel_v3<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_x.data_ptr<float>(), d_y.data_ptr<float>(), N);
    cudaDeviceSynchronize();
}

void reduce_v4(torch::Tensor d_x, torch::Tensor d_y, int N)
{
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    reduce_kernel_v4<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_x.data_ptr<float>(), d_y.data_ptr<float>(), N);
    cudaDeviceSynchronize();
}

void reduce_v5(torch::Tensor d_x, torch::Tensor d_y, int N)
{
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    reduce_kernel_v5<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_x.data_ptr<float>(), d_y.data_ptr<float>(), N);
    cudaDeviceSynchronize();
}

// 注册Python模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("reduce", &reduce, "reduce");
    m.def("reduce_v1", &reduce_v1, "reduce_v1");
    m.def("reduce_v2", &reduce_v2, "reduce_v2");
    m.def("reduce_v3", &reduce_v3, "reduce_v3");
    m.def("reduce_v4", &reduce_v4, "reduce_v4");
    m.def("reduce_v5", &reduce_v5, "reduce_v5");
}