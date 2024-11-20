#include <stdio.h>

# define N 10 // Array size

int main() {
    // Step 1: Declare pointers for host and device memory
    int *h_array, *d_array;

    // Step 2: Allocate memory on host
    h_array = (int *)malloc(N * sizeof(int));
    if (h_array == NULL) {
        printf("Memory allocation failed on host\n");
        return 1;
    }

    // Step 3: Initialize the host array
    for (int i = 0; i< N; i++) {
        h_array[i] = i;
    }

    printf("Host array initialized:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Step 4: Allocate memory on device (GPU)
    cudaMalloc(&d_array, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    int *h_array_copy = (int *)malloc(N * sizeof(int));
    // Copy data from device to host
    
    cudaMemcpy(h_array_copy, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the copied array
    printf("Data copied back from device to host:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_array_copy[i]);
    }
    printf("\n");

    // Step 5: Free memory on host and device
    free(h_array);
    cudaFree(d_array);

    return 0;
}