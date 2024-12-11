#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 2       // Rows of A and C
#define N 3       // Columns of B and C
#define K 4       // Columns of A, rows of B
#define BATCH 3   // Number of matrices in batch

void printBatchedMatrix(float *arr, int rows, int cols, int batch_id, int batch_stride) {
    float* batch_ptr = arr + batch_id * batch_stride;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%.2f ", batch_ptr[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Host matrices
    float *h_A = (float*)malloc(M * K * BATCH * sizeof(float));
    float *h_B = (float*)malloc(K * N * BATCH * sizeof(float));
    float *h_C = (float*)malloc(M * N * BATCH * sizeof(float));

    // Initialize matrices with different values for each batch
    for(int b = 0; b < BATCH; b++) {
        for(int i = 0; i < M * K; i++) {
            h_A[b * M * K + i] = i + b + 1.0f;
        }
        for(int i = 0; i < K * N; i++) {
            h_B[b * K * N + i] = i + b + 1.0f;
        }
    }

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * BATCH * sizeof(float));
    cudaMalloc(&d_B, K * N * BATCH * sizeof(float));
    cudaMalloc(&d_C, M * N * BATCH * sizeof(float));

    // Create arrays of pointers for batched operation
    float **d_Aarray, **d_Barray, **d_Carray;
    float **h_Aarray = (float**)malloc(BATCH * sizeof(float*));
    float **h_Barray = (float**)malloc(BATCH * sizeof(float*));
    float **h_Carray = (float**)malloc(BATCH * sizeof(float*));

    cudaMalloc(&d_Aarray, BATCH * sizeof(float*));
    cudaMalloc(&d_Barray, BATCH * sizeof(float*));
    cudaMalloc(&d_Carray, BATCH * sizeof(float*));

    // Set up array of pointers
    for(int i = 0; i < BATCH; i++) {
        h_Aarray[i] = d_A + i * M * K;
        h_Barray[i] = d_B + i * K * N;
        h_Carray[i] = d_C + i * M * N;
    }

    // Copy matrices and pointer arrays to device
    cudaMemcpy(d_A, h_A, M * K * BATCH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * BATCH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Aarray, h_Aarray, BATCH * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Barray, h_Barray, BATCH * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Carray, h_Carray, BATCH * sizeof(float*), cudaMemcpyHostToDevice);

    // Perform batched matrix multiplication: C = alpha*A*B + beta*C
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemmBatched(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      M, N, K,
                      &alpha,
                      (const float**)d_Aarray, M,
                      (const float**)d_Barray, K,
                      &beta,
                      d_Carray, M,
                      BATCH);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * BATCH * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results for each batch
    for(int b = 0; b < BATCH; b++) {
        printf("\nBatch %d:\n", b);
        printf("Matrix A:\n");
        printBatchedMatrix(h_A, M, K, b, M * K);
        printf("\nMatrix B:\n");
        printBatchedMatrix(h_B, K, N, b, K * N);
        printf("\nResult C = A*B:\n");
        printBatchedMatrix(h_C, M, N, b, M * N);
        printf("\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Aarray);
    cudaFree(d_Barray);
    cudaFree(d_Carray);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Aarray);
    free(h_Barray);
    free(h_Carray);
    cublasDestroy(handle);

    return 0;
}