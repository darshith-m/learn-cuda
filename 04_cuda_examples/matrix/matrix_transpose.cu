#include <cuda_runtime.h>
#include <iostream>
#include <chrono>  // For CPU timing

// Function to verify the transposition result
bool verifyTransposition(const float* h_out, int rows, int cols) {
    bool isCorrect = true;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float expected = j * rows + i;
            float actual = h_out[i * cols + j];
            if (std::abs(expected - actual) > 1e-5) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "Expected " << expected << ", got " << actual << std::endl;
                isCorrect = false;
            }
        }
    }
    return isCorrect;
}

__global__ void transpose_conflict_free(const float* d_in, float* d_out, int rows, int cols) {
    __shared__ float tile[32][32+1];  // Add +1 to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    if (x < cols && y < rows) {
        // Load data into shared memory
        tile[tid_y][tid_x] = d_in[y * cols + x];
    }
    __syncthreads();

    // Write transposed data from shared memory
    x = blockIdx.y * 32 + threadIdx.x;  // Swap block indices
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < rows && y < cols) {
        d_out[y * rows + x] = tile[tid_x][tid_y];
    }
    __syncthreads();
}

// Naive CPU implementation of matrix transposition
void transpose_cpu(const float* h_in, float* h_out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_out[j * rows + i] = h_in[i * cols + j];
        }
    }
}

int main() {
    int rows = 4096;
    int cols = 4096;

    size_t size = rows * cols * sizeof(float);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    float* h_out_cpu = (float*)malloc(size);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            h_in[i * cols + j] = i * cols + j;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);

    // GPU transposition
    cudaEventRecord(startEvent, 0);
    transpose_conflict_free<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Calculate GPU elapsed time
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);

    // CPU transposition
    auto cpuStart = std::chrono::high_resolution_clock::now();
    transpose_cpu(h_in, h_out_cpu, rows, cols);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;

    // Verify the result
    bool transpositionCorrectGpu = verifyTransposition(h_out, rows, cols);
    bool transpositionCorrectCpu = verifyTransposition(h_out_cpu, rows, cols);

    if (transpositionCorrectGpu && transpositionCorrectCpu) {
        std::cout << "Matrix transposition verified successfully!" << std::endl;
    } else {
        std::cout << "Matrix transposition verification failed." << std::endl;
    }

    // Print execution times
    std::cout << "GPU execution time: " << gpuTime << " ms" << std::endl;
    std::cout << "CPU execution time: " << cpuTime.count() << " ms" << std::endl;

    // Calculate speedup
    double speedup = cpuTime.count() / gpuTime;
    std::cout << "Speedup (CPU vs GPU): " << speedup << "x" << std::endl;

    free(h_in);
    free(h_out);
    free(h_out_cpu);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
