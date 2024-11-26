#include <cuda_runtime.h>
#include <iostream>

__global__ void normalSumKernel(int *input, int *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Without atomic operation: Leads to race conditions
        *result += input[idx];
    }
}

__global__ void atomicSumKernel(int *input, int *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform atomic add if within bounds
    if (idx < size) {
        atomicAdd(result, input[idx]);
    }
}

int main() {
    const int size = 1024;
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;

    // Allocate and initialize host memory
    int *h_input = new int[size];
    int h_result_normal = 0;
    int h_result_atomic = 0;

    for (int i = 0; i < size; i++) {
        h_input[i] = 1;
    }

    // Allocate Device Memory
    int *d_input, *d_result_normal, *d_result_atomic;
    cudaMalloc((void **)&d_input, size * sizeof(int));
    cudaMalloc((void **)&d_result_normal, sizeof(int));
    cudaMalloc((void **)&d_result_atomic, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_normal, &h_result_normal, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_atomic, &h_result_atomic, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    normalSumKernel<<<numBlocks, blockSize>>>(d_input, d_result_normal, size);
    atomicSumKernel<<<numBlocks, blockSize>>>(d_input, d_result_atomic, size);

    // Copy result back to host
    cudaMemcpy(&h_result_normal, d_result_normal, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_result_atomic, d_result_atomic, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Sum of array elements (normalSumKernel): "<< h_result_normal << std::endl;
    std::cout << "Sum of array elements (atomicSumKernel): "<< h_result_atomic << std::endl;

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_result_normal);
    cudaFree(d_result_atomic);
    delete[] h_input;

    return 0;
}