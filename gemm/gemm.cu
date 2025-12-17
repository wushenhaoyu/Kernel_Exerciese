#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


__global__ void gemm_kernel_v1(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, const int M, const int N, const int K)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if(tx < M && ty < N)
    {
        float sum = 0.0f;
        for(int k = 0; k < K; k++)
        {
            sum += A[tx * K + k] * B[k * N + ty];
        }
        C[tx * N + ty] = sum;
    }
}

__global__ void gemm_kernel_v2(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, const int M, const int N, const int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = blockDim.x * ty + tx;

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    __shared__ float a_share[BM][BK];
    __shared__ float b_share[BK][BN];

    float c[TM][TN] = {0.0f};

    int a_share_x = (tid & 1) << 2;
    int a_share_y = tid >> 1;
    int b_share_x = (tid & 31) << 2;
    int b_share_y = tid >> 5;

    int a_global_y = bx * BM + a_share_y;
    int b_global_x = by * BN + b_share_x;

    for(int bk = 0; bk < (K + BK - 1 ) / BK ; bk++)
    {
        int a_global_x = bk * BK + a_share_x;
        int b_global_y = bk * BK + b_share_y;
        FLOAT4(a_share[a_share_y][a_share_x]) = FLOAT4(A[a_global_y * K + a_global_x]);
        FLOAT4(b_share[b_share_y][b_share_x]) = FLOAT4(B[b_global_y * N + b_global_x]);
        __syncthreads();
        for(int k = 0; k < BK; k++)
        {
            for(int i = 0; i < TM; i++)
            {
                for(int j = 0; j < TN; j++)
                {
                    c[i][j] +=  a_share[ty * TM + i][k] * b_share[k][tx * TN + j];
                }
            }
        }

        __syncthreads();

        for(int i = 0; i < TM; i++)
        {
            for(int j = 0; j < TN; j++)
            {
                int c_global_x = by * BN + tx * TN + j;
                int c_global_y = bx * BM + ty * TM + i;
                if(c_global_x < N && c_global_y < M)
                {
                    C[c_global_y * N + c_global_x] = c[i][j];
                }
            }
        }
    }
}

// bank_conflict 

__global__ void gemm_kernel_v3(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int M, int N, int K)
{
        int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = blockDim.x * ty + tx;

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    
    const int TM = 8;
    const int TN = 8;

    __shared__ float a_share[BK][BM];
    __shared__ float b_share[BK][BN];

    float c[TM][TN] = {0.0f};

    float a_reg_load[4];

    float a_reg_comp[TM];
    float b_reg_comp[TN];

    int a_share_x = (tid & 1) << 2;
    int a_share_y = tid >> 1;
    int b_share_x = (tid & 31) << 2;
    int b_share_y = tid >> 5;

    int a_global_y = bx * BM + a_share_y;
    int b_global_x = by * BN + b_share_x;

    for(int bk = 0; bk < (K + BK - 1 ) / BK ; bk++)
    {
        int a_global_x = bk * BK + a_share_x;
        int b_global_y = bk * BK + b_share_y;
        FLOAT4(a_reg_load[0]) = FLOAT4(A[a_global_y * K + a_global_x]);
        a_share[a_share_x    ][a_share_y] = a_reg_load[0];
        a_share[a_share_x + 1][a_share_y] = a_reg_load[1];
        a_share[a_share_x + 2][a_share_y] = a_reg_load[2];
        a_share[a_share_x + 3][a_share_y] = a_reg_load[3];
        FLOAT4(b_share[b_share_y][b_share_x]) = FLOAT4(B[b_global_y * N + b_global_x]);
        __syncthreads();
        for(int k = 0; k < BK; k++)
        {
            FLOAT4(a_reg_comp[0]) = FLOAT4(a_share[k][ty * TM / 2]);
            FLOAT4(b_reg_comp[0]) = FLOAT4(b_share[k][tx * TN / 2]);
            FLOAT4(a_reg_comp[4]) = FLOAT4(a_share[k][ty * TM / 2 + BM / 2]);
            FLOAT4(b_reg_comp[4]) = FLOAT4(b_share[k][tx * TN / 2 + BN / 2]);
            for(int i = 0; i < TM; i++)
            {
                for(int j = 0; j < TN; j++)
                {
                    c[i][j] +=  a_reg_comp[i] * b_reg_comp[j];
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TM / 2; i++) {
            int c_global_y = by * BM + ty * TM / 2 + i;
            int c_global_x = bx * BN + tx * TN / 2;
            int c_global_addr = c_global_y * N + c_global_x;
            FLOAT4(C[c_global_addr]) = FLOAT4(c[i][0]);
            FLOAT4(C[c_global_addr + BN / 2]) = FLOAT4(c[i][4]);
        }

        #pragma unroll
        for (int i = 0; i < TM / 2; i++) {
            int c_global_y = by * BM + BM / 2 + ty * TM / 2 + i;
            int c_global_x = bx * BN + tx * TN / 2;
            int c_global_addr = c_global_y * N + c_global_x;
            FLOAT4(C[c_global_addr]) = FLOAT4(c[i + TM / 2][0]);
            FLOAT4(C[c_global_addr + BN / 2]) = FLOAT4(c[i + TM / 2][4]);
        }
    }
}

void gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K) {
    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    for (int i = 0; i < M * N; ++i) {
        C_ptr[i] = 0.0f;
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                C_ptr[m * N + n] += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
        }
    }
}

void gemm_v1(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    gemm_kernel_v1<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

void gemm_v2(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((M + 128 - 1) / 128, (N + 128 - 1) / 128);
    gemm_kernel_v2<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

void gemm_v3(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((M + 128 - 1) / 128, (N + 128 - 1) / 128);
    gemm_kernel_v3<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm", &gemm, "gemm");
    m.def("gemm_v1", &gemm_v1, "gemm_v1");
    m.def("gemm_v2", &gemm_v2, "gemm_v2");
    m.def("gemm_v3", &gemm_v3, "gemm_v3");
}