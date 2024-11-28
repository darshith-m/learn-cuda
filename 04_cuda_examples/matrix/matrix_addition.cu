// Matrix Addition: CPU vs CUDA Implementation
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// GPU kernel for matrix addition
__global__ void matrixAddKernel(int *a, int *b, int *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

// CPU implementation of matrix addition
void matrixAddCPU(int *a, int *b, int *c, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            c[idx] = a[idx] + b[idx];
        }
    }
}

int main() {
    const int rows = 4096;
    const int cols = 4096;
    const int size = rows * cols * sizeof(int);
    
    // Host memory allocation
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c_cpu = (int*)malloc(size);
    int *h_c_gpu = (int*)malloc(size);
    
    // Initialize matrices
    for (int i = 0; i < rows * cols; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }
    
    // Device memory allocation
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, 
                 (rows + blockDim.y - 1) / blockDim.y);
    
    // CPU Matrix Addition using chrono
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixAddCPU(h_a, h_b, h_c_cpu, rows, cols);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU Matrix Addition
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixAddKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, rows, cols);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < rows * cols; i++) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            correct = false;
            break;
        }
    }
    
    printf("Matrix Addition Results:\n");
    printf("CPU Time: %.2f milliseconds\n", cpu_time);
    printf("GPU Time: %.2f milliseconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time/gpu_time);
    printf("Results match: %s\n", correct ? "Yes" : "No");
    
    // Cleanup
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}