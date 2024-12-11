#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Parameters
    int batch_size = 32;
    int in_channels = 3;
    int in_height = 224;
    int in_width = 224;
    int out_channels = 64;
    int kernel_size = 3;
    int padding = 1;
    int stride = 1;

    // Create descriptors
    // Create input tensor descriptor
    cudnnTensorDescriptor_t input_descriptor;     // Declare descriptor handle
    cudnnCreateTensorDescriptor(&input_descriptor); // Initialize descriptor
    cudnnSetTensor4dDescriptor(
        input_descriptor,      // Descriptor handle
        CUDNN_TENSOR_NCHW,    // Memory layout (N=batch, C=channels, H=height, W=width)
        CUDNN_DATA_FLOAT,     // Data type
        batch_size,           // N: Number of images
        in_channels,          // C: Number of channels
        in_height,           // H: Image height 
        in_width             // W: Image width
    );

    // Create filter/kernel descriptor
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(
        kernel_descriptor,     // Descriptor handle  
        CUDNN_DATA_FLOAT,     // Data type
        CUDNN_TENSOR_NCHW,    // Memory layout
        out_channels,         // Number of output feature maps
        in_channels,          // Number of input feature maps
        kernel_size,          // Height of each filter
        kernel_size           // Width of each filter
    );

    // Create convolution descriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(
        convolution_descriptor, // Descriptor handle
        padding, padding,      // Zero-padding height, width
        stride, stride,        // Stride height, width  
        1, 1,                 // Dilation height, width
        CUDNN_CROSS_CORRELATION, // Convolution mode
        CUDNN_DATA_FLOAT       // Math precision
    );
    
    // Calculate output dimensions
    int n, c, h, w;
    cudnnGetConvolution2dForwardOutputDim(
        convolution_descriptor,  // Convolution descriptor
        input_descriptor,       // Input tensor descriptor
        kernel_descriptor,      // Filter descriptor
        &n, &c, &h, &w         // Output dimensions (batch, channels, height, width)
    );

    // Create output tensor descriptor
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(
        output_descriptor,     // Descriptor handle
        CUDNN_TENSOR_NCHW,    // Memory layout
        CUDNN_DATA_FLOAT,     // Data type
        n, c, h, w            // Output dimensions
    );

    // Find best convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf[1];
    int returnedAlgoCount;
    cudnnFindConvolutionForwardAlgorithm(
        cudnn,                 // cuDNN handle
        input_descriptor,      // Input descriptor
        kernel_descriptor,     // Filter descriptor
        convolution_descriptor, // Convolution descriptor
        output_descriptor,     // Output descriptor
        1,                    // Requested algorithm count
        &returnedAlgoCount,   // Returned algorithm count
        algo_perf             // Performance metrics
    );

    // Get required workspace size
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,                 // cuDNN handle
        input_descriptor,      // Input descriptor
        kernel_descriptor,     // Filter descriptor
        convolution_descriptor, // Convolution descriptor
        output_descriptor,     // Output descriptor
        algo_perf[0].algo,    // Selected algorithm
        &workspace_size       // Required workspace size
    );

    // Allocate memory
    void* workspace;
    cudaMalloc(&workspace, workspace_size);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, batch_size * in_channels * in_height * in_width * sizeof(float));
    cudaMalloc(&d_kernel, out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_output, n * c * h * w * sizeof(float));

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(
        cudnn,                 // cuDNN handle
        &alpha,                // Scaling factor for input
        input_descriptor,      // Input tensor descriptor
        d_input,              // Input data pointer
        kernel_descriptor,     // Filter descriptor
        d_kernel,             // Filter data pointer
        convolution_descriptor, // Convolution descriptor
        algo_perf[0].algo,    // Convolution algorithm
        workspace,            // Workspace memory pointer
        workspace_size,       // Workspace size in bytes
        &beta,                // Scaling factor for output
        output_descriptor,     // Output tensor descriptor
        d_output              // Output data pointer
    );

    // Cleanup
    cudaFree(workspace);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}