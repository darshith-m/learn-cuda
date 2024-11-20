#include <stdio.h>

# define N 64 // Total number of threads
# define M 1024 // Total number of threads
# define BLOCK_SIZE 512

// Kernel demonstrating warp-level synchronization
__global__ void warpLevelSync(int *data) {
    int idx = threadIdx.x;
    int warpIdx = idx / 32; // The warp index

    // All threads in the same warp process same data
    int value = data[idx];

    //Synchronize all threads in the warp (implicitly synchronized within a warp)
    value += warpIdx; // Example of independent warp work

    data[idx] = value; // Write back to global memory
}

__global__ void blockLevelSum(int *data, int *blockSums) {
    __shared__ int sharedMem[1024]; // Shared memory for the block

    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Global index
    int localIdx = threadIdx.x;                     // Local index within the block

    // Load data into shared memory
    sharedMem[localIdx] = data[idx];
    __syncthreads(); // Ensure all threads have loaded data

    // Perform parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localIdx < stride) {
            sharedMem[localIdx] += sharedMem[localIdx + stride];
        }
        __syncthreads(); // Ensure all threads complete before next iteration
    }

    // Write the block sum to global memory
    if (localIdx == 0) {
        blockSums[blockIdx.x] = sharedMem[0];
    }
}

__global__ void gridLevelSum(int *blockSums, int iteration) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Perform computation on block sums
    blockSums[idx] += iteration * blockIdx.x;

    // Print the intermediate state
    if (threadIdx.x == 0) {
        printf("Block %d: Updated sum after iteration %d = %d\n", blockIdx.x, iteration, blockSums[idx]);
    }
}


int main() {
    int *h_data; // Host data
    int *d_data; // Device data
    
    h_data = (int *)malloc(N * sizeof(int));
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = 1;
    }

    // Allocate device memory
    cudaMalloc(&d_data, N * sizeof(int));

    // Copy host data to device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // 1. Warp-level synchronization
    printf("Warp-level synchronization:\n");
    warpLevelSync<<<1, N>>>(d_data);

    // Copy data back to host
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");



    // 2. Block-level synchronization
    printf("\nBlock-level synchronization:\n");
    int h_blockSums[M / BLOCK_SIZE];
    int *d_blockSums;

    h_data = (int *)malloc(M * sizeof(int));
    // Initialize the data array
    for (int i = 0; i < M; i++) {
        h_data[i] = 1; // All elements are 1
    }

    // Allocate device memory
    cudaMalloc(&d_data, M * sizeof(int));
    cudaMalloc(&d_blockSums, (M / BLOCK_SIZE) * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_data, h_data, M * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    blockLevelSum<<<M / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, d_blockSums);

    // Copy block sums back to the host
    cudaMemcpy(h_blockSums, d_blockSums, (M / BLOCK_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    printf("Block sums:\n");
    for (int i = 0; i < M / BLOCK_SIZE; i++) {
        printf("Block %d sum: %d\n", i, h_blockSums[i]);
    }

    

    // 3. Grid-level synchronization
    printf("\nGrid-level synchronization:\n");
        // Initialize block sums
    for (int i = 0; i < M / BLOCK_SIZE; i++) {
        h_blockSums[i] = 256; // Each block sum starts at 256
    }

    // Allocate device memory
    cudaMalloc(&d_blockSums, (M / BLOCK_SIZE) * sizeof(int));
    cudaMemcpy(d_blockSums, h_blockSums, (M / BLOCK_SIZE) * sizeof(int), cudaMemcpyHostToDevice);

    // Perform iterative computation with host synchronization
    for (int iteration = 1; iteration <= 5; iteration++) {
        gridLevelSum<<<M / BLOCK_SIZE, BLOCK_SIZE>>>(d_blockSums, iteration);
        cudaDeviceSynchronize(); // Ensure all blocks are complete
    }

    // Copy final results back to the host
    cudaMemcpy(h_blockSums, d_blockSums, (M / BLOCK_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the final results
    printf("Final block sums:\n");
    for (int i = 0; i < M / BLOCK_SIZE; i++) {
        printf("Block %d sum: %d\n", i, h_blockSums[i]);
    }

    // Free memory
    cudaFree(d_data);
    cudaFree(d_blockSums);

    // Free host memory
    free(h_data);

    return 0;
}

