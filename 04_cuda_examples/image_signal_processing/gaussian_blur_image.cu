#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>

__global__ void gaussianBlurColorKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    const float* kernel,
    int kernelSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
    int halfKernel = kernelSize / 2;
    
    for (int ky = -halfKernel; ky <= halfKernel; ky++) {
        for (int kx = -halfKernel; kx <= halfKernel; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            int idx = (py * width + px) * 3;
            float kernelVal = kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
            
            sumB += input[idx] * kernelVal;
            sumG += input[idx + 1] * kernelVal;
            sumR += input[idx + 2] * kernelVal;
        }
    }
    
    int outIdx = (y * width + x) * 3;
    output[outIdx] = static_cast<unsigned char>(sumB);
    output[outIdx + 1] = static_cast<unsigned char>(sumG);
    output[outIdx + 2] = static_cast<unsigned char>(sumR);
}

void gaussianBlurColorCPU(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    const float* kernel,
    int kernelSize
) {
    int halfKernel = kernelSize / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
            
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    int px = std::min(std::max(x + kx, 0), width - 1);
                    int py = std::min(std::max(y + ky, 0), height - 1);
                    int idx = (py * width + px) * 3;
                    float kernelVal = kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                    
                    sumB += input[idx] * kernelVal;
                    sumG += input[idx + 1] * kernelVal;
                    sumR += input[idx + 2] * kernelVal;
                }
            }
            
            int outIdx = (y * width + x) * 3;
            output[outIdx] = static_cast<unsigned char>(sumB);
            output[outIdx + 1] = static_cast<unsigned char>(sumG);
            output[outIdx + 2] = static_cast<unsigned char>(sumR);
        }
    }
}

void createGaussianKernel(float* kernel, int kernelSize, float sigma) {
    int half = kernelSize / 2;
    float sum = 0.0f;
    
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float value = exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            int idx = (y + half) * kernelSize + (x + half);
            kernel[idx] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }
}

int main() {
    // Load image
    cv::Mat image = cv::imread("./image_signal_processing/input.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image\n";
        return -1;
    }
    
    int width = image.cols;
    int height = image.rows;
    int kernelSize = 7;
    float sigma = 5.0f;
    
    // Allocate memory
    size_t imageSize = width * height * 3;
    unsigned char *d_input, *d_output;
    float *d_kernel;
    std::vector<unsigned char> h_output_gpu(imageSize);
    std::vector<unsigned char> h_output_cpu(imageSize);
    std::vector<float> h_kernel(kernelSize * kernelSize);
    
    createGaussianKernel(h_kernel.data(), kernelSize, sigma);
    
    // CUDA allocation and copy
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    
    cudaMemcpy(d_input, image.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    gaussianBlurColorKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_kernel, kernelSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaMemcpy(h_output_gpu.data(), d_output, imageSize, cudaMemcpyDeviceToHost);
    
    // CPU timing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    gaussianBlurColorCPU(image.data, h_output_cpu.data(), width, height, h_kernel.data(), kernelSize);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();
    
    // OpenCV reference
    cv::Mat result_opencv;
    cv::GaussianBlur(image, result_opencv, cv::Size(kernelSize, kernelSize), sigma);
    
    // Validation
    cv::Mat result_gpu(height, width, CV_8UC3, h_output_gpu.data());
    cv::Mat result_cpu(height, width, CV_8UC3, h_output_cpu.data());
    
    cv::Mat diff_gpu_opencv, diff_cpu_opencv;
    cv::absdiff(result_gpu, result_opencv, diff_gpu_opencv);
    cv::absdiff(result_cpu, result_opencv, diff_cpu_opencv);
    
    cv::Scalar gpu_error = cv::mean(diff_gpu_opencv);
    cv::Scalar cpu_error = cv::mean(diff_cpu_opencv);
    
    // Print results
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU Time: " << gpuTime << " ms\n";
    std::cout << "CPU Time: " << cpuTime << " ms\n";
    std::cout << "Speedup: " << cpuTime / gpuTime << "x\n\n";
    
    std::cout << "Validation Results:\n";
    std::cout << "Mean absolute difference per channel (GPU vs OpenCV):\n";
    std::cout << "B: " << gpu_error[0] << " G: " << gpu_error[1] << " R: " << gpu_error[2] << "\n";
    std::cout << "Mean absolute difference per channel (CPU vs OpenCV):\n";
    std::cout << "B: " << cpu_error[0] << " G: " << cpu_error[1] << " R: " << cpu_error[2] << "\n";
    
    // Save results
    cv::imwrite("./image_signal_processing/result_gpu.jpg", result_gpu);
    cv::imwrite("./image_signal_processing/result_cpu.jpg", result_cpu);
    cv::imwrite("./image_signal_processing/result_opencv.jpg", result_opencv);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}