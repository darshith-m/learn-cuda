#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK_SIZE 16

// Simplified GPU kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < M && col < K) {
        // Compute the dot product for the element C[row, col]
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = sum;
    }
}

// CPU implementation
void matrixMulCPU(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

int main() {
    const int M = 1024;  // A rows
    const int N = 1024;  // A cols, B rows
    const int K = 1024;  // B cols
    
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_cpu = (float*)malloc(size_C);
    float *h_C_gpu = (float*)malloc(size_C);
    
    // Initialize matrices
    for (int i = 0; i < M * N; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < N * K; i++) h_B[i] = rand() / (float)RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // CPU Timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_cpu, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU Timing
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((K + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy result back
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Verify results
    float maxError = 0.0f;
    for (int i = 0; i < M * K; i++) {
        maxError = max(maxError, abs(h_C_cpu[i] - h_C_gpu[i]));
    }
    
    printf("Matrix Multiplication Results (%dx%d):\n", M, N);
    printf("CPU Time: %.2f milliseconds\n", cpu_time);
    printf("GPU Time: %.2f milliseconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time/gpu_time);
    printf("Max Error: %e\n", maxError);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}