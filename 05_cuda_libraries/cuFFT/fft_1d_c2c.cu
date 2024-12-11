#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>

#define NX 4  // Number of rows
#define NY 4  // Number of columns

int main() {
    // Create 2D FFT plan for complex-to-complex transform
    cufftHandle plan;
    cufftPlan2d(
        &plan,      // Handle to FFT plan
        NX,         // Size of transform in X dimension
        NY,         // Size of transform in Y dimension
        CUFFT_C2C   // Transform type: Complex to Complex
    );

    // Allocate device memory for input and output
    cufftComplex* d_input;
    cufftComplex* d_output;
    cudaMalloc((void**)&d_input, sizeof(cufftComplex) * NX * NY);   // Input array
    cudaMalloc((void**)&d_output, sizeof(cufftComplex) * NX * NY);  // Output array

    // Execute complex-to-complex FFT transform
    cufftExecC2C(
        plan,           // FFT plan handle
        d_input,        // Input array (complex)
        d_output,       // Output array (complex)
        CUFFT_FORWARD   // Transform direction (CUFFT_FORWARD or CUFFT_INVERSE)
    );

    // Note on memory layout:
    // 2D data is stored in row-major order
    // Index (i,j) maps to i*NY + j in linear memory
    // For C2C transforms, output size equals input size (NX * NY)

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cufftDestroy(plan);

    return 0;
}
