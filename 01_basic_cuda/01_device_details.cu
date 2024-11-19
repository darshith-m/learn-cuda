#include <cuda_runtime.h>
#include <iostream>

int main() {
    // Get the number of CUDA devices
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    // Check for errors
    if (err != cudaSuccess) {
        std::cerr << "\nCUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    // Print the number of CUDA devices
    std::cout << "\nNumber of CUDA devices: " << deviceCount << std::endl;

    // Get the properties of each CUDA device
    for (int i = 0; i < deviceCount; i++) {
        // Get the properties of the current CUDA device
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        // Print the properties of the current CUDA device
        std::cout << "Device #" << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / 1e9 << " GB" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Clock Rate: " << deviceProp.clockRate / 1e6 << " GHz" << std::endl;

        // Additional properties
        std::cout << "  L2 Cache Size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate / 1e3 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;

        std::cout << std::endl;
     }
}