#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for computing the sum using reduction
__global__ void computeSumKernel(const float* d_arr, float* d_sum, int n) {
    __shared__ float sharedSum[256];  // Shared memory for block-wise sum

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localTid = threadIdx.x;

    // Initialize shared memory with input data or 0 if out of bounds
    if (tid < n) {
        sharedSum[localTid] = d_arr[tid];
    } else {
        sharedSum[localTid] = 0;
    }
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localTid < stride) {
            sharedSum[localTid] += sharedSum[localTid + stride];
        }
        __syncthreads();
    }

    // Write the block's sum to global memory
    if (localTid == 0) {
        atomicAdd(d_sum, sharedSum[0]);  // Atomic addition to accumulate sums across blocks
    }
}

int main() {
    int n = 1 << 25;  // Array size (1 million elements)
    size_t size = n * sizeof(float);

    // Host arrays
    std::vector<float> h_arr(n);
    float h_sum = 0.0f;

    // Initialize the array with random values
    for (int i = 0; i < n; ++i) {
        h_arr[i] = static_cast<float>(rand() % 1000);  // Random values between 0 and 1000
    }

    // ========== CPU Implementation ==========
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // Compute the sum on the CPU
    float cpuSum = 0.0f;
    for (size_t i = 0; i < h_arr.size(); ++i) {
        cpuSum += h_arr[i];
    }

    h_sum = cpuSum;

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = end_cpu - start_cpu;

    std::cout << "Sum (CPU): " << h_sum << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " seconds" << std::endl;

    // ========== CUDA Implementation ==========

    // Device arrays
    float *d_arr, *d_sum;
    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_sum, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_arr, h_arr.data(), size, cudaMemcpyHostToDevice);

    // Initialize device sum to 0
    float initSum = 0.0f;
    cudaMemcpy(d_sum, &initSum, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    computeSumKernel<<<gridSize, blockSize>>>(d_arr, d_sum, n);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    std::chrono::duration<float> gpu_duration = end_gpu - start_gpu;

    std::cout << "Sum (CUDA): " << h_sum << std::endl;
    std::cout << "GPU Time: " << gpu_duration.count() << " seconds" << std::endl;

    // Calculate Speedup
    float speedup = cpu_duration.count() / gpu_duration.count();
    std::cout << "Speedup (CPU vs GPU): " << speedup << "x" << std::endl;

    // Free memory
    cudaFree(d_arr);
    cudaFree(d_sum);

    return 0;
}
