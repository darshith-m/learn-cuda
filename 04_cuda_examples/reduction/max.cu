#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <climits>  // For INT_MIN

// CUDA kernel for finding the maximum element
__global__ void findMaxKernel(const int* d_arr, int* d_max, int n) {
    __shared__ int sharedMax[256];  // Shared memory to store block-wise max values
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localTid = threadIdx.x;

    if (tid < n) {
        sharedMax[localTid] = d_arr[tid];
    } else {
        sharedMax[localTid] = INT_MIN; // Use INT_MIN for initialization
    }
    __syncthreads();

    // Perform reduction in the block to find the block's maximum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localTid < stride) {
            sharedMax[localTid] = max(sharedMax[localTid], sharedMax[localTid + stride]);
        }
        __syncthreads();
    }

    // Only the first thread writes the block's max to global memory
    if (localTid == 0) {
        atomicMax(d_max, sharedMax[0]);
    }
}

int main() {
    int n = 1 << 25;  // Array size (1 million elements)
    size_t size = n * sizeof(int);

    // Host arrays
    std::vector<int> h_arr(n);
    int h_max = INT_MIN; // Use INT_MIN for initialization

    // Initialize array with random integer values
    for (int i = 0; i < n; ++i) {
        h_arr[i] = rand() % 1000;  // Random values between 0 and 1000
    }

    // ========== CPU Implementation ==========

    auto start_cpu = std::chrono::high_resolution_clock::now();

    int maxVal = h_arr[0];  // Initialize the maximum with the first element

    // Iterate through the array to find the maximum value
    for (size_t i = 1; i < h_arr.size(); ++i) {
        if (h_arr[i] > maxVal) {
            maxVal = h_arr[i];  // Update the max value when a larger element is found
        }
    }

    h_max = maxVal;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = end_cpu - start_cpu;

    std::cout << "Maximum value (CPU): " << h_max << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " seconds" << std::endl;

    // ========== CUDA Implementation ==========

    // Device arrays
    int *d_arr, *d_max;
    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_max, sizeof(int));
    cudaMemcpy(d_arr, h_arr.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    findMaxKernel<<<gridSize, blockSize>>>(d_arr, d_max, n);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    std::chrono::duration<float> gpu_duration = end_gpu - start_gpu;

    std::cout << "Maximum value (CUDA): " << h_max << std::endl;
    std::cout << "GPU Time: " << gpu_duration.count() << " seconds" << std::endl;

    // Calculate Speedup
    float speedup = cpu_duration.count() / gpu_duration.count();
    std::cout << "Speedup (CPU vs GPU): " << speedup << "x" << std::endl;

    // Free memory
    cudaFree(d_arr);
    cudaFree(d_max);

    return 0;
}
