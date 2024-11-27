#include <iostream>
#include <cuda_runtime.h>

#define N 4096  // Size of the matrix

__global__ void matrixAddRowMajor(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

int main() {
    int size = N * N * sizeof(float);
    
    // Allocate memory for matrices A, B, C
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    
    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;  // Fill with 1
        h_B[i] = 2.0f;  // Fill with 2
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel with appropriate block size
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    cudaEventRecord(start);
    matrixAddRowMajor<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float timeRowMajor;
    cudaEventElapsedTime(&timeRowMajor, start, stop);  // Time in milliseconds
    std::cout << "Row-major matrix addition completed in " << timeRowMajor << " ms" << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
