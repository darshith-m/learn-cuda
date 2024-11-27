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
    const int chunkSize = dataSize / 4; // Breaking the data into 4 chunks
    const int numBlocks = (chunkSize + blockSize - 1) / blockSize;

    // Allocate host and device memory
    float *h_data = new float[dataSize];
    float *d_data[4];  // Using 4 streams for 4 chunks
    cudaStream_t streams[4];

    // Allocate memory for device chunks and create streams
    for (int i = 0; i < 4; ++i) {
        cudaMalloc((void **)&d_data[i], chunkSize * sizeof(float));
        cudaStreamCreate(&streams[i]);
    }

    // Initialize host data
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = static_cast<float>(i);
    }

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Record the start event
    cudaEventRecord(startEvent, 0);

    // Process the data in 4 chunks using streams
    for (int i = 0; i < 4; ++i) {
        int offset = i * chunkSize;

        // Asynchronous memory copy and kernel execution
        cudaMemcpyAsync(d_data[i], h_data + offset, chunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        simpleKernel<<<numBlocks, blockSize, 0, streams[i]>>>(d_data[i], chunkSize);
        cudaMemcpyAsync(h_data + offset, d_data[i], chunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams
    // for (int i = 0; i < 4; ++i) {
    //     cudaStreamSynchronize(streams[i]);
    // }

    // Record the stop event
    cudaEventRecord(stopEvent, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stopEvent);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    // Clean up
    for (int i = 0; i < 4; ++i) {
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    delete[] h_data;

    // Print execution time
    std::cout << "Execution time with streams: " << milliseconds << " ms\n";

    return 0;
}
