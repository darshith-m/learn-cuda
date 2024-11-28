#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(1); \
    } \
}

// CUDA kernel for prefix sum
__global__ void prefixSumKernelSimple(int* d_in, int* d_out, int n) {
    extern __shared__ int sharedData[];  // Shared memory for the block allocated at runtime
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localTid = threadIdx.x;

    // Load input into shared memory
    sharedData[localTid] = (tid < n) ? d_in[tid] : 0;
    __syncthreads();

    // Incremental prefix sum within the block
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (localTid >= stride) {
            temp = sharedData[localTid - stride];
        }
        __syncthreads();  // Wait for all threads before writing
        sharedData[localTid] += temp;
        __syncthreads();  // Wait for all threads after writing
    }

    // Write results to global memory
    if (tid < n) {
        d_out[tid] = sharedData[localTid];
    }
}

// Host function for CPU prefix sum
void cpuPrefixSum(const std::vector<int>& input, std::vector<int>& output) {
    output[0] = input[0];
    for (size_t i = 0; i < input.size(); ++i) {
        output[i+1] = output[i] + input[i+1];
    }
}

// Host function for CUDA prefix sum
void cudaPrefixSum(const std::vector<int>& h_in, std::vector<int>& h_out, float& gpu_time_ms) {
    int n = h_in.size();
    size_t size = n * sizeof(int);

    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice));

    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Create CUDA events to measure the GPU time
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    // Start the GPU timer
    CUDA_CHECK(cudaEventRecord(start_gpu));

    prefixSumKernelSimple<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Stop the GPU timer
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    // Measure GPU time
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    // Clean up CUDA events
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));
}

int main() {
    int n = 1 << 10;  // Use a small array for easy testing
    std::vector<int> h_in(n), h_out_cpu(n), h_out_gpu(n);

    // Initialize input array with random values
    for (int i = 0; i < n; ++i) {
        h_in[i] = rand() % 100;
    }

    // Print input array
    // std::cout << "Input Array: ";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << h_in[i] << " ";
    // }
    // std::cout << std::endl;

    // ========== CPU Prefix Sum ==========
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuPrefixSum(h_in, h_out_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = end_cpu - start_cpu;

    std::cout << "CPU Time: " << cpu_duration.count() << " seconds" << std::endl;

    // Print CPU result
    // std::cout << "CPU Prefix Sum: ";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << h_out_cpu[i] << " ";
    // }
    // std::cout << std::endl;

    // ========== CUDA Prefix Sum ==========
    float gpu_time_ms = 0.0f; // Initialize the variable to hold GPU time
    cudaPrefixSum(h_in, h_out_gpu, gpu_time_ms);  // Pass the variable to update

    // Print GPU result
    // std::cout << "GPU Prefix Sum: ";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << h_out_gpu[i] << " ";
    // }
    // std::cout << std::endl;

    // Verify results
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (h_out_cpu[i] != h_out_gpu[i]) {
            correct = false;
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, h_out_cpu[i], h_out_gpu[i]);
            break;
        }
    }

    if (correct) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    // ========== Speedup Calculation ==========
    // GPU time is in milliseconds. We convert it to seconds
    float gpu_time_sec = gpu_time_ms / 1000.0f;
    std::cout << "GPU Time: " << gpu_time_sec << " seconds" << std::endl;

    // Calculate and print speedup (CPU time / GPU time)
    float speedup = cpu_duration.count() / gpu_time_sec;
    std::cout << "Speedup (CPU / GPU): " << speedup << "x" << std::endl;

    return 0;
}


