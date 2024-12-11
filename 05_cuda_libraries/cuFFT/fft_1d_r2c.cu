#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>

#define N 8  // Size of the input array (number of elements)

int main() {
    // Create FFT plan for 1D real-to-complex transform
    cufftHandle plan;
    cufftPlan1d(
        &plan,     // Handle to FFT plan
        N,         // Size of FFT
        CUFFT_R2C, // Transform type: Real to Complex
        1          // Number of 1D transforms to perform
    );

    // Allocate device memory
    float* d_input;        // Real input array
    cufftComplex* d_output;  // Complex output array
    cudaMalloc((void**)&d_input, sizeof(float) * N);  // For N real elements
    // Output size is N/2 + 1 due to conjugate symmetry in R2C transform
    cudaMalloc((void**)&d_output, sizeof(cufftComplex) * (N/2 + 1));

    // Execute real-to-complex FFT transform
    cufftExecR2C(
        plan,      // FFT plan handle
        d_input,   // Input array (real)
        d_output   // Output array (complex)
    );

    // Cleanup FFT plan
    cufftDestroy(plan);  // Destroys FFT plan and frees associated resources
    cudaFree(d_input);
    cudaFree(d_output);
    // Note on output size:
    // For real-to-complex (R2C) transforms, output size is N/2 + 1
    // This is because FFT of real data has conjugate symmetry,
    // so only half of the complex output needs to be stored
    return 0;
}
