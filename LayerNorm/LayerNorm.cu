#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

__global__ void LayerNorm_kernel_v1(float *out, float *mean, float *rstd, float *input, float *weight, float *bias, int B, int T, int C)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float m = 0.0f;
    for(int c = 0 ; c < C ; c++)
    {
        m += input[tid * C + c];
    }
    m = m / C;
    mean[tid] = m ;
    float v = 0.0f;
    for(int c = 0 ; c < C ; c ++)
    {
        float diff = input[tid * C + c] - m;
        v += diff * diff;
    } 
    v = v / C;

    float s = 1.0f / sqrtf(v + 1e-5f);
    rstd[tid] = s;
    for(int c = 0 ; c < C ; c ++)
    {
        out[tid * C + c] = (input[tid * C + c] - m) * s * weight[c] + bias[c];
    }
}

__global__ void mean_kernel(float* mean, const float* input, int N, int C)
{
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    float sum = 0.0f;
    for(int c = tid ; c < C ; c += blockDim.x)
    {
        sum += input[idx * C + c];
    }
    shared[tid] = sum;
    __syncthreads();
    for(int offset = blockDim.x >> 1 ; offset > 0 ; offset >>= 1)
    {
        if(tid < offset)
        {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        mean[idx] = shared[0] / C;
    }
}

__global__ void rstd_kernel(float* rstd, float* mean, const float* input, int N, int C)
{
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    float sum = 0.0f;
    for(int c = tid ; c < C ; c += blockDim.x)
    {
        float diff = input[idx * C + c] - mean[idx];
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    for(int offset = blockDim.x >> 1 ; offset > 0 ; offset >>= 1)
    {
        if(tid < offset)
        {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

__global__ void LayerNorm_kernel_v2(float *out, float *mean, float *rstd, float *input, float *weight, float *bias, int B, int T, int C)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int bt = tid / C;
    int c = tid % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = input[tid];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[tid] = o;
}

__global__ void LayerNorm_kernel_v3(float *out, float *mean, float *rstd, float *input, float *weight, float *bias, int B, int T, int C, int N)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }
    float* x = input + idx * C;
    float sum = 0.0f;
    for(int c = warp.thread_rank() ; c < C ; c += warp.size())
    {
        sum += x[c];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>());
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    sum = 0.0f;
    for(int c = warp.thread_rank() ; c < C ; c += warp.size())
    {
        float diff = x[c] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>());
    float s = 1.0f / sqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    float* o = out + idx * C;
    for(int c = warp.thread_rank() ; c < C ; c += warp.size())
    {
        float n = s * (x[c] - m);
        __stcs(o + c, n * weight[c] + bias[c]);
    }
}


__global__ void LayerNorm_kernel_v4(float *out, float *mean, float *rstd, float *input, float *weight, float *bias, int B, int T, int C, int N)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }
    float* x = input + idx * C;
    float sum = 0.0f;
    for(int c = warp.thread_rank() ; c < C ; c += warp.size())
    {
        sum += x[c];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>());
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    sum = 0.0f;
    for(int c = warp.thread_rank() ; c < C ; c += warp.size())
    {
        float diff = x[c] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>());
    float s = 1.0f / sqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    float* o = out + idx * C;
    for(int c = warp.thread_rank() ; c < C ; c += warp.size())
    {
        float n = s * (x[c] - m);
        __stcs(o + c, n * weight[c] + bias[c]);
    }
}


__global__ void LayerNorm_kernel_v5(float *out, float *mean, float *rstd, float *input, float *weight, float *bias, int B, int T, int C, int N)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }
    float* x = input + idx * C;
    float sum = 0.0f;
    float sum2 = 0.0f;
    for(int c = warp.thread_rank() ; c < C ; c += warp.size())
    {
        float xi = x[c];
        sum += xi;
        sum2 += xi * xi;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>());
    sum2 = cg::reduce(warp, sum2, cg::plus<float>());
    sum = sum / C;
    sum2 = sum2 / C;
    float m = sum;
    float var = sum2 - sum * sum;
    float s = 1.0f / sqrtf(var + 1e-5f);

    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
        __stcs(rstd + idx, s);
    }
    float* o = out + idx * C;
    for(int c = warp.thread_rank() ; c < C ; c += warp.size())
    {
        float n = s * (x[c] - m);
        __stcs(o + c, n * weight[c] + bias[c]);
    }
}




void LayerNorm(torch::Tensor out, torch::Tensor mean, torch::Tensor rstd, torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{

    

    float eps = 1e-5f;
    int B = input.size(0);  
    int T = input.size(1);  
    int C = input.size(2);  
    for(int b = 0 ; b <  B ; b ++)
    {
        for(int t = 0 ; t < T ; t ++)
        {
            float sum = 0.0f;
            for(int c = 0 ; c < C ; c++)
            {
                sum += input[b][t][c].item<float>();
            }

            float m = sum / C;

            float v = 0.0f;
            for(int c = 0 ; c < C ; c ++)
            {
                float diff = input[b][t][c].item<float>() - m;
                v += diff * diff;
            }

            v = v / C;

            float s = 1.0f / sqrtf(v + eps);

            for(int c = 0 ; c < C ; c ++)
            {
                float n = (input[b][t][c].item<float>() - m) * s;
                out[b][t][c] = n * weight[c].item<float>() + bias[c].item<float>();
            }
            mean[b][t] = m;
            rstd[b][t] = s;
        }
    }
}

void LayerNorm_v1(torch::Tensor out, torch::Tensor mean, torch::Tensor rstd, torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    int B = input.size(0);  
    int T = input.size(1);  
    int C = input.size(2);  
    int N = B * T;
    dim3 blockDim(256);
    dim3 gridDim((N + 255) / 256);
    LayerNorm_kernel_v1<<<gridDim, blockDim>>>(out.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), B, T, C);
}

void LayerNorm_v2(torch::Tensor out, torch::Tensor mean, torch::Tensor rstd, torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    int B = input.size(0);  
    int T = input.size(1);  
    int C = input.size(2);  
    int N = B * T;
    dim3 blockDim(256);
    dim3 gridDim((N + 255) / 256);
    mean_kernel<<<gridDim, blockDim, 256 * sizeof(float)>>>(mean.data_ptr<float>(), input.data_ptr<float>(), N, C);
    rstd_kernel<<<gridDim, blockDim, 256 * sizeof(float)>>>(rstd.data_ptr<float>(), mean.data_ptr<float>(), input.data_ptr<float>(), N, C);
    LayerNorm_kernel_v2<<<gridDim, blockDim>>>(out.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), B, T, C);
}

void LayerNorm_v3(torch::Tensor out, torch::Tensor mean, torch::Tensor rstd, torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    int B = input.size(0);  
    int T = input.size(1);  
    int C = input.size(2);  
    int N = B * T;
    dim3 blockDim(256);
    dim3 gridDim((N + 255) / 256);
    LayerNorm_kernel_v3<<<gridDim, blockDim>>>(out.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), B, T, C, N);
}

void LayerNorm_v4(torch::Tensor out, torch::Tensor mean, torch::Tensor rstd, torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    int B = input.size(0);  
    int T = input.size(1);  
    int C = input.size(2);  
    int N = B * T;
    dim3 blockDim(256);
    dim3 gridDim((N + 255) / 256);
    LayerNorm_kernel_v4<<<gridDim, blockDim>>>(out.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), B, T, C, N);

}

void LayerNorm_v5(torch::Tensor out, torch::Tensor mean, torch::Tensor rstd, torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    int B = input.size(0);  
    int T = input.size(1);  
    int C = input.size(2);  
    int N = B * T;
    dim3 blockDim(256);
    dim3 gridDim((N + 255) / 256);
    LayerNorm_kernel_v5<<<gridDim, blockDim>>>(out.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), B, T, C, N);

}

__global__ void LayerNorm_backward_kernel_v1(float* dinput, float* dweight, float* dbias,
                       const float* dout, const float* input, const float* weight, 
                       const float* mean, const float* rstd,
                       int B, int T, int C) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= B*T) return;
    int b = tid / T;
    int t = tid % T;
    const float* dout_ = dout + (b * T + t) * C;
    const float* input__ = input + (b * T + t) * C;
    const float mean_= mean[b * T + t];
    const float rstd_ = rstd[b * T + t];
    float* dinput_ = dinput + (b * T + t) * C;
    float dout_weight_sum = 0.0f;
    float dout_weight_x__sum = 0.0f;
    for(int c = 0; c < C; c++) {
        float dout_weight = dout_[c];
        float dout_weight_x_ = dout_weight * input__[c];
        dout_weight_sum += dout_weight;
        dout_weight_x__sum += dout_weight_x_;
    }
    dout_weight_sum /= C;
    dout_weight_x__sum /= C;
    for(int c = 0; c < C; c++) {
        float x_ = (input__[c] - mean_) * rstd_;
        float dout_weight = dout_[c];
        dbias[c] += dout_weight;
        dweight[c] += dout_weight * x_;
        dinput_[c] = (dout_weight * weight[c] - (x_ * dout_weight_x__sum + dout_weight_sum)) * rstd_;
    }
}


__global__ void LayerNorm_backward_kernel_v2(float* dinput, float* dweight, float* dbias,
                       const float* dout, const float* input, const float* weight, 
                       const float* mean, const float* rstd,
                       int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) return;

    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = input + b * T * C + t * C;
    float* dinp_bt = dinput + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    #pragma unroll
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        dbias_shared[i] = 0.0f;
        dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>());
    dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>());
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        atomicAdd(&dbias_shared[i], dout_bt[i]);
        atomicAdd(&dweight_shared[i], norm_bti * dout_bt[i]);
        float dval = 0.0f;
        dval += dnorm_i; 
        dval -= dnorm_mean; 
        dval -= norm_bti * dnorm_norm_mean; 
        dval *= rstd_bt; 
        dinp_bt[i] += dval;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        atomicAdd(&dbias[i], dbias_shared[i]);
        atomicAdd(&dweight[i], dweight_shared[i]);
    }
}


void LayerNorm_backward(torch::Tensor dinput, torch::Tensor dweight, torch::Tensor dbias,
                       torch::Tensor dout, torch::Tensor input, torch::Tensor weight, 
                       torch::Tensor mean, torch::Tensor rstd,
                       int B, int T, int C) {

    float* dinput_ptr = dinput.data_ptr<float>();
    float* dweight_ptr = dweight.data_ptr<float>();
    float* dbias_ptr = dbias.data_ptr<float>();
    const float* dout_ptr = dout.data_ptr<float>();
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* mean_ptr = mean.data_ptr<float>();
    const float* rstd_ptr = rstd.data_ptr<float>(); 

    for(int b = 0 ; b < B ; b ++)
    {
        for(int t = 0 ; t < T ; t ++)
        {
            const float* dout_ptr_ = dout_ptr + (b * T + t) * C;
            const float* input_ptr_ = input_ptr + (b * T + t) * C;
            const float mean_ptr_ = mean_ptr[b * T + t];
            const float rstd_ptr_ = rstd_ptr[b * T + t];
            float* dinput_ptr_ = dinput_ptr + (b * T + t) * C;


            float dout_weight_sum = 0.0f;
            float dout_weight_x__sum = 0.0f;
            for(int c = 0 ; c < C ; c ++)
            {
                float x_ = (input_ptr_[c] - mean_ptr_) * rstd_ptr_;
                float dout_weight = weight_ptr[c] * dout_ptr_[c];
                dout_weight_sum += dout_weight;
                dout_weight_x__sum += x_ * dout_weight;
            }
            dout_weight_sum /= C;
            dout_weight_x__sum /= C;
            for(int c = 0 ; c < C ; c ++)
            {
                float x_ = (input_ptr_[c] - mean_ptr_) * rstd_ptr_;
                float dout_weight = weight_ptr[c] * dout_ptr_[c];
                dbias_ptr[c] += dout_ptr_[c];
                dweight_ptr[c] += dout_ptr_[c] * x_;
                dinput_ptr_[c] = (dout_weight - (x_ * dout_weight_x__sum + dout_weight_sum)) * rstd_ptr_;
            }
        }
    }
}

void LayerNorm_backward_v1(torch::Tensor dinput, torch::Tensor dweight, torch::Tensor dbias,
                       torch::Tensor dout, torch::Tensor input, torch::Tensor weight, 
                       torch::Tensor mean, torch::Tensor rstd) {
    int B = input.size(0);  
    int T = input.size(1);  
    int C = input.size(2);  
    int N = B * T;
    dim3 blockDim(256);
    dim3 gridDim((N + 255) / 256);
    LayerNorm_backward_kernel_v1<<<gridDim, blockDim>>>(dinput.data_ptr<float>(), dweight.data_ptr<float>(), dbias.data_ptr<float>(),
                       dout.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(),
                       mean.data_ptr<float>(), rstd.data_ptr<float>(),
                       B, T, C);
}

void LayerNorm_backward_v2(torch::Tensor dinput, torch::Tensor dweight, torch::Tensor dbias,
                       torch::Tensor dout, torch::Tensor input, torch::Tensor weight, 
                       torch::Tensor mean, torch::Tensor rstd) {
    int B = input.size(0);  
    int T = input.size(1);  
    int C = input.size(2);  
    int N = B * T;
    dim3 blockDim(256);
    dim3 gridDim((N + 255) / 256);
    size_t shared_mem_size = 2 * C * sizeof(float); // for dbias_shared and dweight_shared
    LayerNorm_backward_kernel_v2<<<gridDim, blockDim, shared_mem_size>>>(dinput.data_ptr<float>(), dweight.data_ptr<float>(), dbias.data_ptr<float>(),
                       dout.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(),
                       mean.data_ptr<float>(), rstd.data_ptr<float>(),
                       B, T, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { 
    m.def("LayerNorm", &LayerNorm, "LayerNorm");
    m.def("LayerNorm_v1", &LayerNorm_v1, "LayerNorm_v1");
    m.def("LayerNorm_v2", &LayerNorm_v2, "LayerNorm_v2");
    m.def("LayerNorm_v3", &LayerNorm_v3, "LayerNorm_v3");
    m.def("LayerNorm_v4", &LayerNorm_v4, "LayerNorm_v4");
    m.def("LayerNorm_v5", &LayerNorm_v5, "LayerNorm_v5");
    m.def("LayerNorm_backward", &LayerNorm_backward, "LayerNorm_backward");
    m.def("LayerNorm_backward_v1", &LayerNorm_backward_v1, "LayerNorm_backward_v1");
    m.def("LayerNorm_backward_v2", &LayerNorm_backward_v2, "LayerNorm_backward_v2");
}
