#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

__global__ void softmax_kernel_v1(float * __restrict__ input, float * __restrict__ output, const int N, const int C)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < N)
    {
        float max_val = -1e10;
        float sum = 0.0f;
        for(int c = 0; c < C; c++)
        {
            if (input[n * C + c] > max_val)
            {
                sum = sum * expf(max_val - input[n * C + c]) + 1.0f;
                max_val = input[n * C + c];
            }else{
                sum += expf(input[n * C + c] - max_val);
            }
        }

        for(int c = 0; c < C; c++)
        {
            output[n * C + c] = expf(input[n * C + c] - max_val) / sum;
        }
    }
}


struct __align__(8) SumMax
{
    float maxval;
    float sum;
};

// forceinline helps avoid function call overhead
__device__ __forceinline__ SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger_m = a_bigger ? a : b;
    SumMax smaller_m = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger_m.maxval;
    res.sum = bigger_m.sum + smaller_m.sum * expf(smaller_m.maxval - bigger_m.maxval);
    return res;
}
__global__ void softmax_kernel_v2(float * __restrict__ input, float * __restrict__ output, const int N, const int C)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx < N)
    {
        float *x = input + idx * C;
        SumMax sum_max;
        sum_max.maxval = -1e10;
        sum_max.sum = 0.0f;
        for(int i = warp.thread_rank(); i < C ; i += warp.size())
        {
            sum_max = reduce_sum_max_op(sum_max, {x[i], 1.0f});
        }
        SumMax final_sum_max = cg::reduce(warp , sum_max, reduce_sum_max_op);
        for(int i = warp.thread_rank(); i < C ; i += warp.size())
        {
            __stcs(output + idx * C + i, expf(x[i] - final_sum_max.maxval) / final_sum_max.sum);
        }
    }
}


void softmax(torch::Tensor input, torch::Tensor output)
{
    //check input if on cpu
    int N = input.size(0);
    int C = input.size(1);
    float *input_ptr = input.data_ptr<float>();
    float *output_ptr = output.data_ptr<float>();
    for(int n = 0 ; n < N; n++)
    {
        float max_val = -1e10;
        float sum = 0.0f;
        for(int c = 0; c < C; c++)
        {
            if (input_ptr[n * C + c] > max_val)
            {
                sum = sum * expf(max_val - input_ptr[n * C + c]) + 1.0f;
                max_val = input_ptr[n * C + c];
            }else{
                sum += expf(input_ptr[n * C + c] - max_val);
            }
        }

        for(int c = 0; c < C; c++)
        {
            output_ptr[n * C + c] = expf(input_ptr[n * C + c] - max_val) / sum;
        }
    }
}

void softmax_v1(torch::Tensor input, torch::Tensor output)
{
    int N = input.size(0);
    int C = input.size(1);
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    softmax_kernel_v1<<<gridDim, blockDim>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, C);
}

void softmax_v2(torch::Tensor input, torch::Tensor output)
{
    int N = input.size(0);
    int C = input.size(1);
    dim3 blockDim(256);
    dim3 gridDim((N * 32 + blockDim.x - 1) / blockDim.x);
    softmax_kernel_v2<<<gridDim, blockDim>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax", &softmax, "softmax");
    m.def("softmax_v1", &softmax_v1, "softmax_v1");
    m.def("softmax_v2", &softmax_v2, "softmax_v2");
}