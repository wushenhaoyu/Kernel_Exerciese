
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h> 
#include <cuda_runtime.h>
#include <torch/extension.h>
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void gelu_kernel_v1(float *input, float *output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = input[tid];
        float cube = 0.044715f * x * x * x;
        output[tid] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + cube)));
    }
}

__global__ void gelu_kernel_v2(half *__restrict__ input,
                                     half *__restrict__ output,
                                     int N) {
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (tid + 7 < N) {

        __align__(16) half packed_in[8];
        __align__(16) half packed_out[8];

        LDST128BITS(packed_in[0]) = LDST128BITS(input[tid]);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            float xi = __half2float(packed_in[i]);
            float cube = 0.044715f * xi * xi * xi;
            float y = 0.5f * xi *
                      (1.0f + tanhf(0.7978845608028654f * (xi + cube)));
            packed_out[i] = __float2half(y);
        }

        LDST128BITS(output[tid]) = LDST128BITS(packed_out[0]);
    }

    else if (tid < N) {
        for (int i = 0; tid + i < N; i++) {
            float xi = __half2float(input[tid + i]);
            float cube = 0.044715f * xi * xi * xi;
            float y = 0.5f * xi *
                      (1.0f + tanhf(0.7978845608028654f * (xi + cube)));
            output[tid + i] = __float2half(y);
        }
    }
}

__global__ void gelu_backward_kernel_v1(float *input, float *grad_output, float *grad_input, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = input[tid];
        float grad_out = grad_output[tid];
        float tanh_out = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
        float left = 0.5f * (1.0f + tanh_out);
        float right = (1.0f - tanh_out * tanh_out) * (0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x * x));
        grad_input[tid] = grad_out * (left + 0.5f * x * right);
    }
}

__global__ void gelu_backward_kernel_v2(__half *input, __half *grad_output, __half *grad_input, int N) {
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) *  8;
    if(tid < N)
    {
        __half packed_dout[8];
        __half packed_in[8];
        __half packed_dinput[8];
        LDST128BITS(packed_dout[0]) = LDST128BITS(grad_output[tid]);
        LDST128BITS(packed_in[0]) = LDST128BITS(input[tid]);
        for(int i = 0 ; i < 8; i++)
        {
            float xi = (float)packed_in[i];
            float douti = (float)packed_dout[i];
            float cube = 0.044715f * xi * xi * xi;
            float tanh_out = tanhf(0.7978845608028654f * (xi + cube));
            float left = 0.5f * (1.0f + tanh_out);
            float right = (1.0f - tanh_out * tanh_out) * (0.7978845608028654f * (1.0f + 3.0f * 0.044715f * xi * xi));
            float dinput = douti * (left + 0.5f * xi * right);
            packed_dinput[i] = (__half)dinput;
        }

        LDST128BITS(grad_input[tid]) = *reinterpret_cast<float4 *>(&packed_dinput[0]);
    }
}



void gelu(torch::Tensor input, torch::Tensor output) {
    int N = input.numel();
    float *input_ptr = input.data_ptr<float>();
    float *output_ptr = output.data_ptr<float>();
    for (int i = 0; i < N; i++) {
        float x = input_ptr[i];
        float cube = 0.044715f * x * x * x;
        output_ptr[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + cube)));
    }
}

void gelu_v1(torch::Tensor input, torch::Tensor output) {
    int N = input.size(0);
    float *input_ptr = input.data_ptr<float>();
    float *output_ptr = output.data_ptr<float>();
    gelu_kernel_v1<<<(N + 255) / 256, 256>>>(input_ptr, output_ptr, N);
}

void gelu_v2(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "input must be half");
    TORCH_CHECK(output.scalar_type() == torch::kHalf, "output must be half");

    int N = input.numel();

    at::Half* input_ptr  = input.data_ptr<at::Half>();
    at::Half* output_ptr = output.data_ptr<at::Half>();

    int threads = 256;
    int blocks  = (N + threads * 8 - 1) / (threads * 8);

    gelu_kernel_v2<<<blocks, threads>>>(
        reinterpret_cast<half*>(input_ptr),
        reinterpret_cast<half*>(output_ptr),
        N
    );
}


void gelu_backward_v1(torch::Tensor input, torch::Tensor grad_output, torch::Tensor grad_input) {
    int N = input.size(0);
    float *input_ptr = input.data_ptr<float>();
    float *grad_output_ptr = grad_output.data_ptr<float>();
    float *grad_input_ptr = grad_input.data_ptr<float>();
    gelu_backward_kernel_v1<<<(N + 255) / 256, 256>>>(input_ptr, grad_output_ptr, grad_input_ptr, N);
}

/*void gelu_backward_v2(torch::Tensor input, torch::Tensor grad_output, torch::Tensor grad_input, int N) {
    torch::Tensor input_half = input.to(torch::kHalf);
    torch::Tensor grad_output_half = grad_output.to(torch::kHalf);
    torch::Tensor grad_input_half = grad_input.to(torch::kHalf);
    __half *input_ptr = input_half.data_ptr<__half>();
    __half *grad_output_ptr = grad_output_half.data_ptr<__half>();
    __half *grad_input_ptr = grad_input_half.data_ptr<__half>();
    gelu_backward_kernel_v2<<<(N + 255) / 256, 256>>>(input_ptr, grad_output_ptr, grad_input_ptr, N);
    grad_input.copy_(grad_input_half.to(torch::kFloat));
    input.copy_(input_half.to(torch::kFloat));
    grad_output.copy_(grad_output_half.to(torch::kFloat));
}*/


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu", &gelu, "gelu");
    m.def("gelu_v1", &gelu_v1, "gelu_v1");
    m.def("gelu_v2", &gelu_v2, "gelu_v2");
    m.def("gelu_backward_v1", &gelu_backward_v1, "gelu_backward_v1");
   /// m.def("gelu_backward_v2", &gelu_backward_v2, "gelu_backward_v2");
}
