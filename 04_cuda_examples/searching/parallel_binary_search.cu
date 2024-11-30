#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

// CPU Binary Search
int binarySearch(const std::vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// CUDA Kernel for parallel binary search
__global__ void binarySearchKernel(const int* arr, const int* targets, int* results, 
                                 int arrSize, int numSearches) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numSearches) return;
    
    int target = targets[tid];
    int left = 0;
    int right = arrSize - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            results[tid] = mid;
            return;
        }
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    results[tid] = -1;
}

int main() {
    // Parameters
    const int arraySize = 1000000;  // Number of elements in the array
    const int numSearches = 100000; // Number of elements to search in the array
    const int blockSize = 256;
    
    // Generate sorted array
    std::vector<int> arr(arraySize);
    for (int i = 0; i < arraySize; i++) {
        arr[i] = i * 2; // Ensuring sorted array
    }
    
    // Generate random search targets
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, arraySize * 2);
    std::vector<int> targets(numSearches);
    for (int i = 0; i < numSearches; i++) {
        targets[i] = dis(gen);
    }
    
    // CPU Binary Search
    std::vector<int> cpuResults(numSearches);
    auto cpuStart = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numSearches; i++) {
        cpuResults[i] = binarySearch(arr, targets[i]);
    }
    
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart);
    
    // GPU Binary Search Setup
    int *d_arr, *d_targets, *d_results;
    cudaMalloc(&d_arr, arraySize * sizeof(int));
    cudaMalloc(&d_targets, numSearches * sizeof(int));
    cudaMalloc(&d_results, numSearches * sizeof(int));
    
    cudaMemcpy(d_arr, arr.data(), arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets.data(), numSearches * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel
    int numBlocks = (numSearches + blockSize - 1) / blockSize;
    cudaEventRecord(start);
    
    binarySearchKernel<<<numBlocks, blockSize>>>(d_arr, d_targets, d_results, arraySize, numSearches);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get GPU results
    std::vector<int> gpuResults(numSearches);
    cudaMemcpy(gpuResults.data(), d_results, numSearches * sizeof(int), cudaMemcpyDeviceToHost);
    
    float gpuMilliseconds;
    cudaEventElapsedTime(&gpuMilliseconds, start, stop);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < numSearches; i++) {
        if (cpuResults[i] != gpuResults[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": CPU = " << cpuResults[i] 
                      << ", GPU = " << gpuResults[i] << std::endl;
            break;
        }
    }
    
    // Print timing results
    std::cout << "CPU Time: " << cpuDuration.count() << " microseconds" << std::endl;
    std::cout << "GPU Time: " << gpuMilliseconds * 1000 << " microseconds" << std::endl;
    std::cout << "Speedup: " << (float)cpuDuration.count() / (gpuMilliseconds * 1000) << "x" << std::endl;
    std::cout << "Results are " << (correct ? "correct" : "incorrect") << std::endl;
    
    // Cleanup
    cudaFree(d_arr);
    cudaFree(d_targets);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}