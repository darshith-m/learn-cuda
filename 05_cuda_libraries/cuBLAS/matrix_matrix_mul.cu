#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 3   // Rows of A and C
#define K 4   // Columns of A, rows of B
#define N 2   // Columns of B and C

void printMatrix(float *arr, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%.2f ", arr[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Host matrices
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    for(int i = 0; i < M * K; i++) h_A[i] = i + 1.0f;
    for(int i = 0; i < K * N; i++) h_B[i] = i + 1.0f;

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication: C = alpha*A*B + beta*C
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                d_A, M,    // Leading dimension of A
                d_B, K,    // Leading dimension of B
                &beta,
                d_C, M);   // Leading dimension of C

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("Matrix A (%dx%d):\n", M, K);
    printMatrix(h_A, M, K);
    printf("\nMatrix B (%dx%d):\n", K, N);
    printMatrix(h_B, K, N);
    printf("\nResult C = A*B (%dx%d):\n", M, N);
    printMatrix(h_C, M, N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cublasDestroy(handle);

    return 0;
}