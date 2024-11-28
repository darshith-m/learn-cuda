#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

// Utility function to print a small portion of the array for verification
void printArray(const std::vector<int>& data, int n) {
    for (int i = 0; i < std::min(n, 10); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "...\n";
}

// CPU Up-Sweep (Reduction) Phase
void upSweepCPU(std::vector<int>& data) {
    int n = data.size();
    int powerOf2 = 1;
    while (powerOf2 < n) powerOf2 *= 2;
    for (int stride = 1; stride < powerOf2; stride *= 2) {
        for (int i = stride-1; i < n; i += 2 * stride) {
            data[i + stride] += data[i];
        }
    }
}

// CPU Down-Sweep Phase
void downSweepCPU(std::vector<int>& data) {
    int n = data.size();
    int powerOf2 = 1;
    while (powerOf2 < n) powerOf2 *= 2;
    for (int stride = powerOf2 / 2; stride > 0; stride /= 2) {
        for (int i = stride - 1; i < n; i += 2 * stride) {
            int temp = data[i + stride];
            data[i + stride] += data[i];
            data[i] = temp;
        }
    }
}

// CPU Prefix Sum
void prefixSumCPU(std::vector<int>& data) {
    upSweepCPU(data);
    data[data.size() - 1] = 0;  // Set the last element to 0 for the final result
    downSweepCPU(data);
}

// GPU Up-Sweep (Reduction) Phase
__global__ void upSweepGPU(int* d_data, int n) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;
    int globalTid = tid + blockIdx.x * blockDim.x;

    if (globalTid < n) {
        sharedData[tid] = d_data[globalTid];
    } else {
        sharedData[tid] = 0; // Pad with zeros if necessary
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            sharedData[index + 2 * stride - 1] += sharedData[index + stride - 1];
        }
        __syncthreads();
    }

    if (globalTid < n) {
        d_data[globalTid] = sharedData[tid];
    }
}

// GPU Down-Sweep Phase
__global__ void downSweepGPU(int* d_data, int n) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;
    int globalTid = tid + blockIdx.x * blockDim.x;

    if (globalTid == n-1) {
        d_data[globalTid] = 0;
    }
    
    if (globalTid < n) {
        sharedData[tid] = d_data[globalTid];
    } else {
        sharedData[tid] = 0; // Pad with zeros if necessary
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            int temp = sharedData[index + stride - 1];
            sharedData[index + stride - 1] = sharedData[index + 2 * stride - 1];
            sharedData[index + 2 * stride - 1] += temp;
        }
        __syncthreads();
    }

    if (globalTid < n) {
        d_data[globalTid] = sharedData[tid];
    }
}

// GPU Prefix Sum
void prefixSumGPU(int* d_data, int n) {
    // Define block size and shared memory size
    const int blockSize = 1024;  // Maximum number of threads per block
    const int sharedMemSize = blockSize * sizeof(int);  // Shared memory size for each block

    // Calculate the number of blocks needed for the given data size (rounding up)
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the up-sweep (reduction) phase
    upSweepGPU<<<numBlocks, blockSize, sharedMemSize>>>(d_data, n);
    cudaDeviceSynchronize();  // Synchronize to ensure up-sweep is complete

    // Debugging: Copy d_data back to the host and print it
    // std::vector<int> debugData(n);
    // cudaMemcpy(debugData.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Data after up-sweep (first kernel):\n";
    // for (int i = 0; i < std::min(n, 16); ++i) {  // Print the first 10 elements for brevity
    //     std::cout << debugData[i] << " ";
    // }
    // std::cout << "...\n";

    // Launch the down-sweep phase
    downSweepGPU<<<numBlocks, blockSize, sharedMemSize>>>(d_data, n);
    cudaDeviceSynchronize();  // Synchronize to ensure down-sweep is complete
}


int main() {
    // Input size
    int n = 1024;  // Change this to adjust the size (e.g., 1 << 10 for smaller sizes)

    // Create input for CPU and GPU
    std::vector<int> h_data(n, 1); // All ones for simplicity

    // --- CPU Prefix Sum ---
    std::vector<int> h_data_cpu = h_data;  // Copy data for CPU
    auto start = std::chrono::high_resolution_clock::now();
    prefixSumCPU(h_data_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = end - start;

    std::cout << "CPU time taken: " << cpu_duration.count() << " seconds\n";
    printArray(h_data_cpu, 10);  // Print the first 10 elements for verification

    // --- GPU Prefix Sum ---
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    prefixSumGPU(d_data, n);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, startEvent, stopEvent);

    cudaMemcpy(h_data.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU time taken: " << gpu_time / 1000.0f << " seconds\n";  // Convert ms to seconds
    printArray(h_data, 10);  // Print the first 10 elements for verification

    // --- Speedup Calculation ---
    float speedup = cpu_duration.count() / (gpu_time / 1000.0f);  // Convert GPU time from ms to seconds
    std::cout << "Speedup: " << speedup << "x\n";

    // Cleanup
    cudaFree(d_data);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
