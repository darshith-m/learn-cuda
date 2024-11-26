#include <stdio.h>

// Kernel that deliberately introduces an error
__global__ void faultyKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Deliberate out-of-bounds access to global memory
    if (idx >= 1024) {
        data[idx] = idx * 2; // This will cause an illegal memory access
    }
}

// Function to check for CUDA errors
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError(); // Check for the last error
    if (err != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE); // Exit if an error occurred
    }
}

int main() {
    const int N = 1024;
    int *d_data;

    // Allocate device memory
    cudaError_t allocError = cudaMalloc(&d_data, N * sizeof(int));
    if (allocError != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(allocError));
        return EXIT_FAILURE;
    }

    // Launch kernel
    printf("Launching kernel...\n");
    faultyKernel<<<2, 512>>>(d_data);
    checkCudaError("Kernel execution failed"); // Check for kernel launch errors

    // Synchronize device to catch runtime errors
    cudaDeviceSynchronize();
    checkCudaError("CUDA synchronization failed");

    // Free device memory
    cudaFree(d_data);
    checkCudaError("Device memory free failed");

    printf("Program finished successfully.\n");
    return 0;
}
