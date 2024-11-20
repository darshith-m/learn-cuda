#include <stdio.h>

// Kernel to demonstrate memory hierarchy
__global__ void memoryHierarchyDemo(int *globalData, int *output) {
    // Compute thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 1. Register Memory: Variables stored in registers
    int regValue = globalData[idx]; // Load value into a register
    regValue += 10; // Perform some computation

    // 2. Shared Memory: Accessible to all threads within a block
    __shared__ int sharedMem[128]; // Shared memory declaration
    sharedMem[threadIdx.x] = regValue; // Store value into shared memory
    __syncthreads(); // Ensure all threads in the block load shared memory

    // Example: Sum of values in shared memory by first thread in the block
    if (threadIdx.x == 0) {
        int sharedSum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sharedSum += sharedMem[i];
        }
        printf("Block %d: Shared memory sum = %d\n", blockIdx.x, sharedSum);
    }

    // 3. Local Memory: Per-thread memory for dynamically allocated arrays
    int *localMem = (int *)malloc(sizeof(int)); // Dynamically allocated local memory
    *localMem = regValue * 2; // Store computed value in local memory

    // 4. Write Results Back to Global Memory
    output[idx] = *localMem; // Write value from local memory to global memory

    // Free local memory
    free(localMem);
}

int main() {
    const int N = 128; // Total number of threads
    const int BLOCK_SIZE = 32; // Threads per block
    const int GRID_SIZE = N / BLOCK_SIZE; // Number of blocks

    // Host data
    int h_globalData[N], h_output[N];
    for (int i = 0; i < N; i++) {
        h_globalData[i] = i; // Initialize global data with indices
    }

    // Device memory pointers
    int *d_globalData, *d_output;

    // Allocate device memory
    cudaMalloc(&d_globalData, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_globalData, h_globalData, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    memoryHierarchyDemo<<<GRID_SIZE, BLOCK_SIZE>>>(d_globalData, d_output);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output
    printf("Output from GPU:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_globalData);
    cudaFree(d_output);

    return 0;
}
