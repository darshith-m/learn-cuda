#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// Utility function to generate random points
std::vector<float> generateRandomPoints(int numPoints, int dims) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> points(numPoints * dims);
    for (int i = 0; i < points.size(); i++) {
        points[i] = dis(gen);
    }
    return points;
}

// CPU implementation of KNN search
void knnSearchCPU(const std::vector<float>& points, const std::vector<float>& queries,
                  int numPoints, int numQueries, int dims, int k,
                  std::vector<int>& indices) {
    
    std::vector<float> distances(numPoints);
    
    // Iterating through queriess
    for (int q = 0; q < numQueries; q++) {
        // Calculate distances for current query
        for (int p = 0; p < numPoints; p++) {
            float dist = 0.0f;
            for (int d = 0; d < dims; d++) {
                float diff = points[p * dims + d] - queries[q * dims + d]; // distance in dimension d (x or y or z)
                dist += diff * diff; // dist = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2. Square root is omitted as it is unncessary calculation
            }
            distances[p] = dist; // store the distance calculated
        }
        
        // Find k nearest neighbors using partial sort
        std::vector<int> idx(numPoints); // Vector to store indices of points from query
        std::iota(idx.begin(), idx.end(), 0); // Fills indices for all the points from query
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), // sort first k indices based on their distances
            [&distances](int i1, int i2) { return distances[i1] < distances[i2]; });
        
        // Store results
        for (int i = 0; i < k; i++) {
            indices[q * k + i] = idx[i];
        }
    }
}

// CUDA kernel for KNN search
__global__ void knnSearchKernel(const float* points, const float* queries,
                               int numPoints, int dims, int k,
                               float* distances, int* indices) {
    int queryIdx = blockIdx.x;
    int pointIdx = threadIdx.x + blockDim.x * blockIdx.y;
    
    if (pointIdx < numPoints) {
        float dist = 0.0f;
        for (int d = 0; d < dims; d++) {
            float diff = points[pointIdx * dims + d] - queries[queryIdx * dims + d];
            dist += diff * diff;
        }
        distances[queryIdx * numPoints + pointIdx] = dist;
        indices[queryIdx * numPoints + pointIdx] = pointIdx;
    }
}

// Helper function for GPU partial sort
__device__ void insertionSort(float* distances, int* indices, int k) {
    for (int i = 1; i < k; i++) {
        float tempDist = distances[i];
        int tempIdx = indices[i];
        int j = i - 1;
        
        while (j >= 0 && distances[j] > tempDist) {
            distances[j + 1] = distances[j];
            indices[j + 1] = indices[j];
            j--;
        }
        
        distances[j + 1] = tempDist;
        indices[j + 1] = tempIdx;
    }
}

// CUDA kernel for finding k smallest elements
__global__ void findKNearest(float* distances, int* indices,
                            int numPoints, int k,
                            float* kDistances, int* kIndices) {
    int queryIdx = threadIdx.x + blockDim.x * blockIdx.x;
    
    float* queryDistances = distances + queryIdx * numPoints;
    int* queryIndices = indices + queryIdx * numPoints;
    
    // Initialize first k elements
    float localDist[100];  // Assuming k <= 100
    int localIdx[100];
    
    for (int i = 0; i < k; i++) {
        localDist[i] = queryDistances[i];
        localIdx[i] = queryIndices[i];
    }
    insertionSort(localDist, localIdx, k);
    
    // Process remaining elements
    for (int i = k; i < numPoints; i++) {
        if (queryDistances[i] < localDist[k-1]) {
            localDist[k-1] = queryDistances[i];
            localIdx[k-1] = queryIndices[i];
            insertionSort(localDist, localIdx, k);
        }
    }
    
    // Store results
    for (int i = 0; i < k; i++) {
        kDistances[queryIdx * k + i] = localDist[i];
        kIndices[queryIdx * k + i] = localIdx[i];
    }
}

// Main function to run KNN search on GPU
int knnSearchGPU(const std::vector<float>& points, const std::vector<float>& queries,
                  int numPoints, int numQueries, int dims, int k,
                  std::vector<int>& indices) {
    
    // Allocate GPU memory
    float *d_points, *d_queries, *d_distances, *d_kDistances;
    int *d_indices, *d_kIndices;
    
    cudaMalloc(&d_points, numPoints * dims * sizeof(float));
    cudaMalloc(&d_queries, numQueries * dims * sizeof(float));
    cudaMalloc(&d_distances, numQueries * numPoints * sizeof(float));
    cudaMalloc(&d_indices, numQueries * numPoints * sizeof(int));
    cudaMalloc(&d_kDistances, numQueries * k * sizeof(float));
    cudaMalloc(&d_kIndices, numQueries * k * sizeof(int));
    
    // Copy data to GPU
    cudaMemcpy(d_points, points.data(), numPoints * dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, queries.data(), numQueries * dims * sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksY = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(numQueries, blocksY);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch kernels
    knnSearchKernel<<<gridDim, threadsPerBlock>>>(d_points, d_queries, numPoints, dims, k,
                                                 d_distances, d_indices);
    findKNearest<<<1, numQueries>>>(d_distances, d_indices, numPoints, k,
                                   d_kDistances, d_kIndices);
    
    // Record stop event and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back to CPU
    cudaMemcpy(indices.data(), d_kIndices, numQueries * k * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_points);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_indices);
    cudaFree(d_kDistances);
    cudaFree(d_kIndices);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    // Parameters
    const int numPoints = 100000;
    const int numQueries = 1000;
    const int dims = 3; // X,y,z dimensions
    const int k = 10; // Number of nearest neighbors to store
    
    // Generate random points and queries
    auto points = generateRandomPoints(numPoints, dims);
    auto queries = generateRandomPoints(numQueries, dims);
    
    // Vectors to store results
    std::vector<int> cpuIndices(numQueries * k); // For every point, store k number of neighbors
    std::vector<int> gpuIndices(numQueries * k);
    
    // CPU timing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    knnSearchCPU(points, queries, numPoints, numQueries, dims, k, cpuIndices);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count();
    
    // GPU timing handled within knnSearchGPU function
    auto gpuTime = knnSearchGPU(points, queries, numPoints, numQueries, dims, k, gpuIndices);
    
    // Verify results
    int correct = 0;
    for (int i = 0; i < numQueries * k; i++) {
        if (cpuIndices[i] == gpuIndices[i]) correct++;
    }
    
    // Print results
    std::cout << "CPU Time: " << cpuTime << " ms\n";
    std::cout << "GPU Time: " << gpuTime << " ms\n";
    std::cout << "Speedup: " << (float)cpuTime / gpuTime << "x\n";
    std::cout << "Correctness: " << (float)correct / (numQueries * k) * 100 << "%\n";
    
    return 0;
}