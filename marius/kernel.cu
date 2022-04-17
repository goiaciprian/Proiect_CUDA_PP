#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core.hpp>

#include <stdio.h>
#include <math.h>
#include <iostream>

__global__ void test2D(double* m1, double* m2, double* m3, int N)
{
    double PI = 3.14;
    int id_i = threadIdx.x + blockDim.x * blockIdx.x;
    int id_j = threadIdx.y + blockDim.y * blockIdx.y;

    if (id_i < N && id_j < N) {
        m1[id_i * N + id_j] = pow(sin((2 * PI * id_i) / N), 2) + pow(cos((2 * PI * id_j) / N), 2);
        m2[id_i * N + id_j] = pow(sin((2 * PI * id_i) / N), 2) + pow(cos((2 * PI * id_j) / N), 2);
    }
}

__global__ void test1D(double* m1, double* m2, double* m3, int N)
{
    double PI = 3.14;
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < N * N) {
        int id_i = id / N;
        int id_j = id % N;
        m1[id_i * N + id_j] = pow(sin((2 * PI * id_i) / N), 2) + pow(cos((2 * PI * id_j) / N), 2);
        m2[id_i * N + id_j] = pow(sin((2 * PI * id_i) / N), 2) + pow(cos((2 * PI * id_j) / N), 2);
    }
}

__global__ void addMatrix1D(double* m1, double* m2, double* m3, int N) {
    int thread = threadIdx.x;
    int block = blockIdx.x;

    int id = block * blockDim.x + thread;
    if (id < N * N) m3[id] = m1[id] + m2[id];
}

int main()
{
    int N = 1024;

    double* m1_h = new double[N * N];
    double* m2_h = new double[N * N];
    double* m3_h = new double[N * N];

    double* m1_d;
    double* m2_d;
    double* m3_d;

    cudaMalloc((void**)&m1_d, N * N * sizeof(double));
    cudaMalloc((void**)&m2_d, N * N * sizeof(double));
    cudaMalloc((void**)&m3_d, N * N * sizeof(double));

    cudaMemcpy(m1_d, m1_h, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m2_d, m2_h, N * N * sizeof(double), cudaMemcpyHostToDevice);
    

    dim3 threadsPerBlock(16, 32);
    dim3 blocks2D(N / 16, N / 32);
    dim3 blocks1D(N * N / 512 + 1, 512); // 512 = 16*32

    test1D << <blocks1D, threadsPerBlock >> > (m1_d, m2_d, m3_d, N);

    addMatrix1D << <blocks1D, threadsPerBlock >> > (m1_d, m2_d, m3_d, N);

    cudaMemcpy(m3_h, m3_d, N * N * sizeof(double), cudaMemcpyDeviceToHost);


    for (int i = 0; i < N; i++)
    {
        /*
        std::cout<<m3_h[0]<<std::endl;
        std::cout << m3_h[1] << std::endl;
        */
        std::cout << m3_h[i] << std::endl;
    }
}