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
    float *d_data1, *d_data2, *d_data3, *d_data4; // Device chunks
    cudaStream_t stream1, stream2, stream3, stream4;

    // Allocate memory for device chunks
    cudaMalloc((void **)&d_data1, chunkSize * sizeof(float));
    cudaMalloc((void **)&d_data2, chunkSize * sizeof(float));
    cudaMalloc((void **)&d_data3, chunkSize * sizeof(float));
    cudaMalloc((void **)&d_data4, chunkSize * sizeof(float));

    // Create streams
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

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

    // Process the first chunk using stream1
    cudaMemcpyAsync(d_data1, h_data, chunkSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
    simpleKernel<<<numBlocks, blockSize, 0, stream1>>>(d_data1, chunkSize);
    cudaMemcpyAsync(h_data, d_data1, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);

    // Process the second chunk using stream2
    cudaMemcpyAsync(d_data2, h_data + chunkSize, chunkSize * sizeof(float), cudaMemcpyHostToDevice, stream2);
    simpleKernel<<<numBlocks, blockSize, 0, stream2>>>(d_data2, chunkSize);
    cudaMemcpyAsync(h_data + chunkSize, d_data2, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream2);

    // Process the third chunk using stream3
    cudaMemcpyAsync(d_data3, h_data + 2 * chunkSize, chunkSize * sizeof(float), cudaMemcpyHostToDevice, stream3);
    simpleKernel<<<numBlocks, blockSize, 0, stream3>>>(d_data3, chunkSize);
    cudaMemcpyAsync(h_data + 2 * chunkSize, d_data3, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream3);

    // Process the fourth chunk using stream4
    cudaMemcpyAsync(d_data4, h_data + 3 * chunkSize, chunkSize * sizeof(float), cudaMemcpyHostToDevice, stream4);
    simpleKernel<<<numBlocks, blockSize, 0, stream4>>>(d_data4, chunkSize);
    cudaMemcpyAsync(h_data + 3 * chunkSize, d_data4, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream4);

    // Synchronize all streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);

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
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
    cudaFree(d_data4);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    delete[] h_data;

    // Print execution time
    std::cout << "Execution time with streams: " << milliseconds << " ms\n";

    return 0;
}
