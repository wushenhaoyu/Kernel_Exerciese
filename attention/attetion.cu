#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;
#define INFINITY 1e20



__global__ void attention_query_key_kernel(float *preatt, float  *input, int B, int T, int C, int NH)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < B * T * T * NH)
    {
        int t2 = tid % T;
        int t1 = (tid / T) % T;
        if(t2 > t1)
        {
            preatt[tid] = -INFINITY;
            return;
        }
        int h = (tid / (T * T)) % NH;
        int b = tid / (T * T * NH);
        int C3 = C * 3;
        int hs = C / NH;
        float *q = input + b * C3 * T + C3 * t1 + h * hs;
        float *k = input + b * C3 * T + C3 * t2 + h * hs + C;
        float score = 0.0f;
        for(int c = 0; c < hs ; c ++)
        {
            score += q[c] * k[c];
        }
        score *= 1.0 / sqrtf(hs);
        preatt[tid] = score;

    }
}


__global__ void attention_softmax_kernel(float *att, float *preatt, int B, int T, int NH)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < B * T * NH)
    {
        int t1 = tid % T;
        int h = (tid / T) % NH;
        int b = tid / (T * NH);
        float *preatt_ptr = preatt + b * NH * T * T + h * T * T + t1 * T;
        float *att_ptr = att + b * NH * T * T + h * T * T + t1 * T;
        float max_score = -10000.0f;
        for(int t2 = 0 ; t2 < T ; t2 ++)
        {
            float score = preatt_ptr[t2];
            if(score > max_score) max_score = score;
        }
        float expsum = 0.0f;
        for(int t2 = 0 ; t2 < T ; t2 ++)
        {
            float score = expf(preatt_ptr[t2] - max_score);
            att_ptr[t2] = score;
            expsum += score;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
        for(int t2 = 0 ; t2 < T ; t2 ++)
        {
            if(t2 <= t1)
            {
                att_ptr[t2] *= expsum_inv;
            }else{
                att_ptr[t2] = 0.0f;
            }
        }
    }
}

__global__ void attention_value_kernel(float *out, float *att ,float *input, int B, int T, int C, int NH)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < B * T * NH)
    {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        int C3 = C*3;
        int hs = C / NH; // head size

        float* out_ptr = out + b * T * C + t * C + h * hs;
        float* att_ptr = att + b*NH*T*T + h*T*T + t*T;
        for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
        for (int t2 = 0; t2 < T; t2++) {
            float *v = input_ptr + b * C3 * T + C3 * t2 + h * hs + C * 2;
            float att_ptr_cache = att_ptr[t2];
            for (int c = 0; c < hs; c++) {
                out_ptr[c] += att_ptr_cache * v[c];
            }
        }
    }
}
void attention_forward_cpu(Torch::Tensor out, Torch::Tensor preatt, Torch::Tensor att,
                       Torch::Tensor input) 
{
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int B = input.size(0);
    int T = input.size(1);
    int C = input.size(2) / 3;
    int NH = att.size(1);
    float *out_ptr = out.data_ptr<float>();
    float *preatt_ptr = preatt.data_ptr<float>();
    float *att_ptr = att.data_ptr<float>();
    float *input_ptr = input.data_ptr<float>();

    int C3 = C*3;
    int hs = C / NH; //head_size
    float scale = 1.0 / sqrtf(hs);

    for(int b = 0; b < B; b++)
    {
        for(int t = 0 ; t < T ; t ++)
        {
            for(int h = 0; h < NH; h++)
            {
                float *q = input_ptr + b * C3 * T + C3 * t + h * hs;
                float *preatt_ptr_ = preatt_ptr + b * NH * T * T + h * T * T + t * T;
                float *att_ptr_ = att_ptr + b * NH * T * T + h * T * T + t * T;

                float max_score = -10000.0f;
                for(int t2 = 0 ; t2 < T ; t2 ++)
                {
                    float *k = input_ptr + b * C3 * T + C3 * t2 + h * hs + C;
                    float score = 0.0f;
                    for(int c = 0; c < hs ; c ++)
                    {
                        score += q[c] * k[c];
                    }
                    score *= scale;
                    if(score > max_score) max_score = score;
                    preatt_ptr_[t2] = score;

                }

                for(int t2 = t ; t2 < T ; t2 ++)
                {
                    preatt_ptr_[t2] = -INFINITY;
                }

                float expsum = 0.0f;
                for(int t2 = 0 ; t2 < T ; t2 ++)
                {
                    float score = expf(preatt_ptr_[t2] - max_score);
                    att_ptr_[t2] = score;
                    expsum += score;
                }

                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                for(int t2 = 0 ; t2 < T ; t2 ++)
                {
                    if(t2 <= t)
                    {
                        att_ptr_[t2] *= expsum_inv;
                    }else{
                        att_ptr_[t2] = 0.0f;
                    }

                }

                float *out_ptr_ = out_ptr + b * C * T + t * C + h * hs;
                for(int c = 0; c < hs ; c ++) 
                {
                    out_ptr_[c] = 0.0f;
                }
                for(int t2 = 0 ; t2 < T ; t2 ++)
                {
                    float *v = input_ptr + b * C3 * T + C3 * t2 + h * hs + C * 2;
                    float att_ptr_cache = att_ptr_[t2];
                    for(int c = 0; c < hs ; c ++)
                    {
                        out_ptr_[c] += att_ptr_cache * v[c];
                    }
                }

            }
            }
        }
    }
}

void attetion_forward_v1(Torch::Tensor out, Torch::Tensor preatt, Torch::Tensor att, Torch::Tensor input)
{
    int B = input.size(0);
    int T = input.size(1);
    int C = input.size(2) / 3;
    int NH = att.size(1);
    float *out_ptr = out.data_ptr<float>();
    float *preatt_ptr = preatt.data_ptr<float>();
    float *att_ptr = att.data_ptr<float>();
    float *input_ptr = input.data_ptr<float>();
    int total_threads = B * T * T * NH;
    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    attetion_query_key_kernel<<<num_blocks, threads_per_block>>>(preatt_ptr, input_ptr, B, T, C, NH);
    cudaDeviceSynchronize();
    num_blocks = (B * T * NH + threads_per_block - 1) / threads_per_block;
    attention_value_kernel<<<num_blocks, threads_per_block>>>(out_ptr, att_ptr, input_ptr, B, T, C, NH);
    cudaDeviceSynchronize();
    attention_value_kernel<<<num_blocks, threads_per_block>>>(out_ptr, att_ptr, input_ptr, B, T, C, NH);
    cudaDeviceSynchronize();

}


