#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(1); \
    } \
}

// Kernel for block-level prefix sum
__global__ void blockPrefixSumKernel(int* d_in, int* d_out, int* d_blockSums, int n) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localTid = threadIdx.x;

    // Load input into shared memory
    sharedData[localTid] = (tid < n) ? d_in[tid] : 0;
    __syncthreads();

    // Compute prefix sum within block
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (localTid >= stride) {
            temp = sharedData[localTid - stride];
        }
        __syncthreads();
        sharedData[localTid] += temp;
        __syncthreads();
    }

    // Write block results
    if (tid < n) {
        d_out[tid] = sharedData[localTid];
    }

    // Last thread in block writes block sum
    if (localTid == blockDim.x - 1) {
        d_blockSums[blockIdx.x] = sharedData[localTid];
    }
}

// Kernel to add block prefix sums from prvious blocks
__global__ void addBlockSumsKernel(int* d_out, int* d_blockSums, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n && blockIdx.x > 0) {
        d_out[tid] += d_blockSums[blockIdx.x - 1];
    }
}

void cpuPrefixSum(const std::vector<int>& input, std::vector<int>& output) {
    output[0] = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i-1] + input[i];
    }
}

void cudaPrefixSum(const std::vector<int>& h_in, std::vector<int>& h_out, float& gpu_time_ms) {
    int n = h_in.size();
    size_t size = n * sizeof(int);

    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t blockSumsSize = gridSize * sizeof(int);

    int *d_in, *d_out, *d_blockSums;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMalloc(&d_blockSums, blockSumsSize));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Step 1: Compute block-level prefix sums
    blockPrefixSumKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(
        d_in, d_out, d_blockSums, n);

    // Step 2: Compute prefix sum of block sums (using CPU for simplicity)
    std::vector<int> h_blockSums(gridSize);
    CUDA_CHECK(cudaMemcpy(h_blockSums.data(), d_blockSums, blockSumsSize, cudaMemcpyDeviceToHost));
    
    // Adding block sums to blocks 1 to gridSize - 1
    // Block 0 doesnt need this as it doesnt have any previous blocks to sup from
    for (int i = 1; i < gridSize; i++) {
        h_blockSums[i] += h_blockSums[i-1];
    }
    
    CUDA_CHECK(cudaMemcpy(d_blockSums, h_blockSums.data(), blockSumsSize, cudaMemcpyHostToDevice));

    // Step 3: Add block prefix sums back to results
    addBlockSumsKernel<<<gridSize, blockSize>>>(d_out, d_blockSums, n);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Peforming inclusive scan in thrust to verify results
void cudaThrustPrefixSum(const std::vector<int>& h_in, std::vector<int>& h_out, float& gpu_time_ms) {
    thrust::device_vector<int> d_in(h_in);
    thrust::device_vector<int> d_out(h_in.size());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::inclusive_scan(d_in.begin(), d_in.end(), d_out.begin());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    thrust::copy(d_out.begin(), d_out.end(), h_out.begin());

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int n = 1 << 25;
    std::vector<int> h_in(n), h_out_cpu(n), h_out_gpu(n), h_out_thrust(n);

    for (int i = 0; i < n; ++i) {
        h_in[i] = rand() % 10;
    }

    // CPU Prefix Sum
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuPrefixSum(h_in, h_out_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = end_cpu - start_cpu;

    // Custom CUDA Prefix Sum
    float gpu_time_ms = 0.0f;
    cudaPrefixSum(h_in, h_out_gpu, gpu_time_ms);

    // Thrust Prefix Sum
    float thrust_time_ms = 0.0f;
    cudaThrustPrefixSum(h_in, h_out_thrust, thrust_time_ms);

    // Verify results with thrust
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (h_out_thrust[i] != h_out_cpu[i] || h_out_thrust[i] != h_out_gpu[i]) {
            correct = false;
            printf("Mismatch at %d: Thrust = %d, CPU = %d, GPU = %d\n", 
                i, h_out_thrust[i], h_out_cpu[i], h_out_gpu[i]);
            break;
        }
    }
    
    std::cout << "Results " << (correct ? "Results match!" : "do not match!") << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " seconds" << std::endl;
    std::cout << "Custom GPU Time: " << gpu_time_ms/1000 << " seconds" << std::endl;
    std::cout << "Speed-up: "<< cpu_duration.count()/(gpu_time_ms/1000) << "x" << std::endl;
    
    return 0;
}