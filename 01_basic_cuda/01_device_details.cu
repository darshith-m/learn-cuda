#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "\nCUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    std::cout << "\nNumber of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "\nDevice #" << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads Dimension: [" \
                  << deviceProp.maxThreadsDim[0] << ", " \
                  << deviceProp.maxThreadsDim[1] << ", " \
                  << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Size: [" \
                  << deviceProp.maxGridSize[0] << ", " \
                  << deviceProp.maxGridSize[1] << ", " \
                  << deviceProp.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
    }
}