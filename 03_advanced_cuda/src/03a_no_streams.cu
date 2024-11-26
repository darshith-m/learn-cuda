#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * data[idx]; // Example operation: square the value
    }
}

int main() {
    const int dataSize = 1 << 20; // 1 million elements
    const int blockSize = 256;
    const int numBlocks = (dataSize + blockSize - 1) / blockSize;

    // Allocate host and device memory
    float *h_data = new float[dataSize];
    float *d_data;

    cudaMalloc((void **)&d_data, dataSize * sizeof(float));

    // Initialize host data
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = i;
    }

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Record the start event
    cudaEventRecord(startEvent, 0);

    // Copy data to device, run kernel, and copy back to host
    cudaMemcpy(d_data, h_data, dataSize * sizeof(float), cudaMemcpyHostToDevice);
    simpleKernel<<<numBlocks, blockSize>>>(d_data, dataSize);
    cudaMemcpy(h_data, d_data, dataSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Record the stop event
    cudaEventRecord(stopEvent, 0);

    // Wait for the event to complete
    cudaEventSynchronize(stopEvent);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    // Clean up
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_data);
    delete[] h_data;

    // Print execution time
    std::cout << "Execution time without streams: " << milliseconds << " ms\n";

    return 0;
}
