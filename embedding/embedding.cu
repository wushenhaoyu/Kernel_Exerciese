#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;


__global__ void endcoder_kernel_v1(float *out, int *input, float *wte, float *wpe, int B, int T, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;
    if (idx < N)
    {
        int b = idx / T;
        int t = idx % T;
        int index = input[b * T + t];
        float *wte_ptr = wte + index * C;
        float *wpe_ptr = wpe + t * C;
        for (int c = 0; c < C; c++)
        {
            out[b * T * C + t * C + c] = wte_ptr[c] + wpe_ptr[c];
        }
    }
}

__global__ void endcoder_kernel_v2(float *out, int *input, float *wte, float *wpe, int B, int T, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx < N)
    {
        int b = idx / (T * C);
        int t = (idx % (T * C)) / C;
        int c = idx % C;
        int index = input[b * T + t];
        float *wte_ptr = wte + index * C;
        float *wpe_ptr = wpe + t * C;
        out[idx] = wte_ptr[c] + wpe_ptr[c];
    }
}



__global__ void encoder_backward_kernel_v1(float *dwte, float *dwpe,
                                          float *dout, int *input, int B, int T, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx < N)
    {
        int b = idx / (T * C);
        int t = (idx % (T * C)) / C;
        int c = idx % C;
        int index = input[b * T + t];
        float *dwte_ptr = dwte + index * C;
        float *dwpe_ptr = dwpe + t * C;
        dwte_ptr[c] += dout[b * T * C + t * C + c];
        dwpe_ptr[c] += dout[b * T * C + t * C + c];
    }
}


void encoder(torch::Tensor out,
                   torch::Tensor input, torch::Tensor wte, torch::Tensor wpe)
{
    int B = input.size(0);
    int T = input.size(1);
    int C = wte.size(1);
    float *out_ptr = out.data_ptr<float>();
    int *input_ptr = input.data_ptr<int>();
    float *wte_ptr = wte.data_ptr<float>();
    float *wpe_ptr = wpe.data_ptr<float>();
    for(int b = 0 ; b < B ; b ++)
    {
        for(int t = 0 ; t < T ; t ++)
        {
            int index = input[b][t].item<int>();
            float *wte_ptr_ = wte_ptr + index * C;
            float *wpe_ptr_ = wpe_ptr + t * C;
            for(int c = 0 ; c < C ; c ++)
            {
                out_ptr[b * T * C + t * C + c] = wte_ptr_[c] + wpe_ptr_[c];
            }
        }
    }
}

void encoder_v1(torch::Tensor out,
                   torch::Tensor input, torch::Tensor wte, torch::Tensor wpe)
{
    int B = input.size(0);
    int T = input.size(1);
    int C = wte.size(1);
    float *out_ptr = out.data_ptr<float>();
    int *input_ptr = input.data_ptr<int>();
    float *wte_ptr = wte.data_ptr<float>();
    float *wpe_ptr = wpe.data_ptr<float>();
    int threads = 256;
    int blocks = (B * T + threads - 1) / threads;
    endcoder_kernel_v1<<<blocks, threads>>>(out_ptr, input_ptr, wte_ptr, wpe_ptr, B, T, C);
}


void encoder_v2(torch::Tensor out,
                   torch::Tensor input, torch::Tensor wte, torch::Tensor wpe)
{
    int B = input.size(0);
    int T = input.size(1);
    int C = wte.size(1);
    float *out_ptr = out.data_ptr<float>();
    int *input_ptr = input.data_ptr<int>();
    float *wte_ptr = wte.data_ptr<float>();
    float *wpe_ptr = wpe.data_ptr<float>();
    int threads = 256;
    int blocks = (B * T * C + threads - 1) / threads;
    endcoder_kernel_v2<<<blocks, threads>>>(out_ptr, input_ptr, wte_ptr, wpe_ptr, B, T, C);
}

void encoder_backward(torch::Tensor dwte, torch::Tensor dwpe,
                            torch::Tensor dout, torch::Tensor input)
{
    int B = input.size(0);
    int T = input.size(1);
    int C = dwte.size(1);
    float *dwte_ptr = dwte.data_ptr<float>();
    float *dwpe_ptr = dwpe.data_ptr<float>();
    float *dout_ptr = dout.data_ptr<float>();
    int *input_ptr = input.data_ptr<int>();
    for(int b = 0 ; b < B ; b ++)
    {
        for(int t = 0 ; t < T ; t ++)
        {
            int index = input_ptr[b * T + t];
            float *dwte_ptr_ = dwte_ptr + index * C;
            float *dwpe_ptr_ = dwpe_ptr + t * C;
            for(int c = 0 ; c < C ; c ++)
            {
                dwte_ptr_[c] += dout_ptr[b * T * C + t * C + c];
                dwpe_ptr_[c] += dout_ptr[b * T * C + t * C + c];
            }
        }
    }

}



void encoder_backward_v1(torch::Tensor dwte, torch::Tensor dwpe,
                            torch::Tensor dout, torch::Tensor input)
{
    int B = input.size(0);
    int T = input.size(1);
    int C = dwte.size(1);
    float *dwte_ptr = dwte.data_ptr<float>();
    float *dwpe_ptr = dwpe.data_ptr<float>();
    float *dout_ptr = dout.data_ptr<float>();
    int *input_ptr = input.data_ptr<int>();
    int threads = 256;
    int blocks = (B * T * C + threads - 1) / threads;
    encoder_backward_kernel_v1<<<blocks, threads>>>(dwte_ptr, dwpe_ptr, dout_ptr, input_ptr, B, T, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { 
    m.def("encoder", &encoder, "Encoder forward");
    m.def("encoder_v1", &encoder_v1, "Encoder forward v1");
    m.def("encoder_v2", &encoder_v2, "Encoder forward v2");
    m.def("encoder_backward", &encoder_backward, "Encoder backward");
    m.def("encoder_backward_v1", &encoder_backward_v1, "Encoder backward v1");
}