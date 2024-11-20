#include <stdio.h>

// Kernel function to run on GPU
__global__ void printThreadInfo() {
    
    // Get the block number in the grid
    int blockId = blockIdx.x +blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    // Get the starting position of block in global thread index space
    int blockOffset = blockId * blockDim.x * blockDim.y * blockDim.z;
    // Get the thread number in the block
    int threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // Get the global thread id
    int globalThreadId = blockOffset + threadId;
    
    printf("blockId: %d, blockOffset: %d, threadId: %d, globalThreadId: %d\n", blockId, blockOffset, threadId, globalThreadId);
}

int main() {
    // Define the dimensions of the grid and block
    dim3 threadsPerBlock(2, 2, 2); // 2x2x2 threads per block
    dim3 numberOfBlocks(2, 2, 2); // 2x2x2 blocks in the grid

    //Launch the kernel
    printThreadInfo<<<numberOfBlocks, threadsPerBlock>>>();

    // Wait for the GPU to finish before accessing the results:
    cudaDeviceSynchronize();

    return 0;
}