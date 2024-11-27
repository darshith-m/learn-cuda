#include <cuda_runtime.h>
#include <stdio.h>

#define N 4096 * 4096  // Total number of elements
#define THREADS_PER_BLOCK 256

__global__ void noWarpDivergenceExample(int *d_out, int *d_in) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {  // Ensure we don't access out-of-bounds
        // Warp divergence is avoided by predicating on the condition
        int mask = d_in[idx] % 2 == 0;
        // Random logic to just add 1 for even numbers and subtract 1 for odd numbers
        d_out[idx] = (mask) ? (d_in[idx] + 1) : (d_in[idx] - 1);
    }
}

int main() {
    int *h_in = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));
    int *d_in, *d_out;

    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_in[i] = (i % 2 == 0) ? i : -i;  // Mix of positive and negative numbers
    }

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel launch
    dim3 blockSize(THREADS_PER_BLOCK);
    dim3 gridSize((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    cudaEventRecord(start);
    noWarpDivergenceExample<<<gridSize, blockSize>>>(d_out, d_in);
    cudaEventRecord(stop);

    // Wait for the events to complete
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time taken without warp divergence: %.2f ms\n", elapsedTime);

    // Cleanup
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
