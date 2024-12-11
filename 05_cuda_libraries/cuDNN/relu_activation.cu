#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // Initialize cuDNN library
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Define dimensions
    int batch_size = 128;
    int channels = 64;
    int height = 32;
    int width = 32;

    // Create input/output tensor descriptor
    // Both input and output have same dimensions for ReLU
    cudnnTensorDescriptor_t tensor_descriptor;
    cudnnCreateTensorDescriptor(&tensor_descriptor);
    cudnnSetTensor4dDescriptor(
        tensor_descriptor,
        CUDNN_TENSOR_NCHW,     // Memory format: batch-channel-height-width
        CUDNN_DATA_FLOAT,      // Data type of the tensor
        batch_size,            // Number of images in batch
        channels,              // Number of feature maps
        height,                // Height of each feature map
        width                  // Width of each feature map
    );

    // Create activation descriptor
    // Specifies the type and parameters of activation function
    cudnnActivationDescriptor_t activation_descriptor;
    cudnnCreateActivationDescriptor(&activation_descriptor);
    cudnnSetActivationDescriptor(
        activation_descriptor,
        CUDNN_ACTIVATION_RELU,  // ReLU activation function
        CUDNN_NOT_PROPAGATE_NAN, // How to handle NaN values
        0.0                     // ReLU ceiling (not used for standard ReLU)
    );

    // Allocate GPU memory for input and output
    float *d_input, *d_output;
    size_t tensor_size = batch_size * channels * height * width * sizeof(float);
    cudaMalloc(&d_input, tensor_size);   // Input tensor memory
    cudaMalloc(&d_output, tensor_size);  // Output tensor memory

    // Perform ReLU activation
    float alpha = 1.0f;  // Scaling factor for input
    float beta = 0.0f;   // Scaling factor for output
    cudnnActivationForward(
        cudnn,
        activation_descriptor,    // Activation function parameters
        &alpha,                  // Input scaling factor
        tensor_descriptor,       // Input tensor descriptor
        d_input,                // Input data pointer
        &beta,                  // Output scaling factor
        tensor_descriptor,       // Output tensor descriptor
        d_output                // Output data pointer
    );

    // Optional: Perform backward pass for ReLU
    // dy = gradient from next layer
    float *d_dy, *d_dx;
    cudaMalloc(&d_dy, tensor_size);  // Incoming gradient
    cudaMalloc(&d_dx, tensor_size);  // Output gradient

    cudnnActivationBackward(
        cudnn,
        activation_descriptor,    // Activation function parameters
        &alpha,                  // Input scaling factor
        tensor_descriptor,       // Output tensor descriptor (forward pass)
        d_output,               // Output from forward pass
        tensor_descriptor,       // Gradient tensor descriptor
        d_dy,                   // Incoming gradient
        tensor_descriptor,       // Input tensor descriptor (forward pass)
        d_input,                // Input from forward pass
        &beta,                  // Output scaling factor
        tensor_descriptor,       // Output gradient tensor descriptor
        d_dx                    // Output gradient
    );

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_dy);
    cudaFree(d_dx);
    cudnnDestroyTensorDescriptor(tensor_descriptor);
    cudnnDestroyActivationDescriptor(activation_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}