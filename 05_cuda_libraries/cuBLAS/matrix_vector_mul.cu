#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 3   // Number of rows
#define N 3   // Number of columns

void printMatrix(float *arr, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%.2f ", arr[i * cols + j]);
        }
        printf("\n");
    }
}

void printVector(float *arr, int size) {
    for(int i = 0; i < size; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Host arrays
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(M * sizeof(float));

    // Initialize matrix A and vector x
    for(int i = 0; i < M * N; i++) {
        h_A[i] = i + 1.0f;
    }
    for(int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
    }

    // Device arrays
    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix-vector multiplication: y = alpha*A*x + beta*y
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemv(handle, CUBLAS_OP_N, 
                M, N,
                &alpha,
                d_A, M,    // Leading dimension of A
                d_x, 1,    // Stride of x
                &beta,
                d_y, 1);   // Stride of y

    // Copy result back to host
    cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("Matrix A:\n");
    printMatrix(h_A, M, N);
    printf("\nVector x:\n");
    printVector(h_x, N);
    printf("\nResult y = A*x:\n");
    printVector(h_y, M);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);
    cublasDestroy(handle);

    return 0;
}