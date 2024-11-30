#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

void printArray(const std::vector<int>& data, int n) {
    for (int i = 0; i < std::min(n,10); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "...\n";
}

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

void prefixSumCPU(std::vector<int>& data) {
    upSweepCPU(data);
    data[data.size() - 1] = 0;
    downSweepCPU(data);
}

__global__ void blockPrefixSumKernel(int* d_data, int* d_blockSums, int n) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalTid = tid + bid * blockDim.x;

    // Load data into shared memory
    sharedData[tid] = (tid < n) ? d_data[globalTid] : 0;
    __syncthreads();

    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            sharedData[index] += sharedData[index - stride];
        }
        __syncthreads();
    }

    // Store block sum before down sweep
    if (tid == blockDim.x - 1) {
        if (d_blockSums != nullptr) {
            d_blockSums[bid] = sharedData[tid];
        }
        sharedData[tid] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            int temp = sharedData[index];
            sharedData[index] += sharedData[index - stride];
            sharedData[index - stride] = temp;
        }
        __syncthreads();
    }

    // Write results back to global memory
    if (globalTid < n) {
        d_data[globalTid] = sharedData[tid];
    }
}

__global__ void aggregateBlockSums(int* d_blockSums, int numBlocks) {
    int tid = threadIdx.x;
    
    for(int offset = 1; offset < numBlocks; offset *= 2) {
        if(tid >= offset && tid < numBlocks) {
            int temp = d_blockSums[tid - offset];
            __syncthreads();
            d_blockSums[tid] += temp;
        }
        __syncthreads();
    }
}

__global__ void addBlockSums(int* d_data, const int* d_blockSums, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n && blockIdx.x > 0) {
        d_data[tid] += d_blockSums[blockIdx.x - 1];
    }
}

void prefixSumGPU(int* d_data, int n, float& gpu_time) {
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    const int blockSize = 1024;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    const int sharedMemSize = blockSize * sizeof(int);

    // Allocate memory for block sums
    int* d_blockSums;
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));

    // Perform block-level prefix sums
    blockPrefixSumKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_data, d_blockSums, n);
    
    if (numBlocks > 1) {
        // aggregateBlockSums<<<1, numBlocks>>>(d_blockSums, numBlocks);
        //cudaDeviceSynchronize();
        //  Compute prefix sum of block sums (using CPU for simplicity)
        std::vector<int> h_blockSums(numBlocks);
        cudaMemcpy(h_blockSums.data(), d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Adding block sums to blocks 1 to gridSize - 1
        // Block 0 doesnt need this as it doesnt have any previous blocks to sup from
        for (int i = 1; i < numBlocks; i++) { 
            h_blockSums[i] += h_blockSums[i-1]; 
        }
        
        cudaMemcpy(d_blockSums, h_blockSums.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);
        // Add block sums back to each element
        addBlockSums<<<numBlocks, blockSize>>>(d_data, d_blockSums, n);
    }

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&gpu_time, startEvent, stopEvent);

    cudaFree(d_blockSums);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

int main() {
    int n = 1<<25;  // Test with 1M elements
    std::vector<int> h_data(n, 1);

    // CPU implementation
    std::vector<int> h_data_cpu = h_data;
    auto start = std::chrono::high_resolution_clock::now();
    prefixSumCPU(h_data_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = end - start;

    // Custom GPU implementation
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    float gpu_time;
    prefixSumGPU(d_data, n, gpu_time);

    std::vector<int> h_data_gpu(n);
    cudaMemcpy(h_data_gpu.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Thrust implementation for verification
    thrust::device_vector<int> d_data_thrust(h_data);
    
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
    
    thrust::exclusive_scan(d_data_thrust.begin(), d_data_thrust.end(), d_data_thrust.begin());
    
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float thrust_time;
    cudaEventElapsedTime(&thrust_time, startEvent, stopEvent);

    std::vector<int> h_data_thrust(n);
    thrust::copy(d_data_thrust.begin(), d_data_thrust.end(), h_data_thrust.begin());

    // Verify results against Thrust
    // Verify results
    bool results_match = true;
    for(int i = 0; i < n; i++) {
        if(h_data_gpu[i] != h_data_thrust[i] || h_data_cpu[i] != h_data_thrust[i]) {
            printf("Mismatch at %d: Thrust = %d, CPU = %d, GPU = %d\n", 
                i, h_data_thrust[i], h_data_cpu[i], h_data_gpu[i]);
            results_match = false;
            break;
        }
    }
    std::cout << "\nResults match reference implementation: " << (results_match ? "Yes" : "No") << "\n";
    
    // Print timing results
    std::cout << "CPU time: " << cpu_duration.count() << " seconds\n";
    std::cout << "GPU time: " << gpu_time / 1000.0f << " seconds\n";
    std::cout << "Speedup: " << cpu_duration.count()/(gpu_time/1000.0f) << "x\n";

    // std::cout << "\nFirst 10 elements from each implementation:\n";
    // std::cout << "CPU: "; printArray(h_data_cpu, 0);
    // std::cout << "GPU: "; printArray(h_data_gpu, 0);
    // std::cout << "Thrust: "; printArray(h_data_thrust, 0);


    // Cleanup
    cudaFree(d_data);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}