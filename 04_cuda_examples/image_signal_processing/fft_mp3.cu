#include <cuda_runtime.h>
#include <cufft.h>
#include <sndfile.h>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>

const double PI = 3.14159265358979323846;

void cpu_fft_manual(float* input, std::complex<float>* output, int N) {
    for (int i = 0; i < N; i++) {
        int j = 0;
        int k = i;
        for (int l = 0; l < log2(N); l++) {
            j = (j << 1) | (k & 1);
            k = k >> 1;
        }
        if (j > i) {
            output[i] = std::complex<float>(input[j], 0.0f);
            output[j] = std::complex<float>(input[i], 0.0f);
        } else if (i == j) {
            output[i] = std::complex<float>(input[i], 0.0f);
        }
    }

    for (int s = 1; s <= log2(N); s++) {
        int m = 1 << s;
        float angle = -2.0f * PI / m;
        std::complex<float> wm(cosf(angle), sinf(angle));
        
        for (int k = 0; k < N; k += m) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < m/2; j++) {
                std::complex<float> t = w * output[k + j + m/2];
                std::complex<float> u = output[k + j];
                output[k + j] = u + t;
                output[k + j + m/2] = u - t;
                w *= wm;
            }
        }
    }
}

__global__ void bitReversalKernel(float* input, cuComplex* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int j = 0;
        int k = idx;
        for (int l = 0; l < log2f(N); l++) {
            j = (j << 1) | (k & 1);
            k = k >> 1;
        }
        if (j >= idx) {
            output[idx].x = input[j];
            output[idx].y = 0.0f;
            if (j != idx) {
                output[j].x = input[idx];
                output[j].y = 0.0f;
            }
        }
    }
}

__global__ void fftKernel(cuComplex* data, int N, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = 1 << s;
    
    if (idx < N/2) {
        int k = idx / (m/2) * m;
        int j = idx % (m/2);
        
        float angle = -2.0f * PI * j / m;
        cuComplex wm = make_cuComplex(cosf(angle), sinf(angle));
        
        cuComplex u = data[k + j];
        cuComplex t = cuCmulf(wm, data[k + j + m/2]);
        
        data[k + j] = cuCaddf(u, t);
        data[k + j + m/2] = cuCsubf(u, t);
    }
}

void gpu_fft_manual(float* input, std::complex<float>* output, int N) {
    float *d_input;
    cuComplex *d_output;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(cuComplex));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    bitReversalKernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
    
    for (int s = 1; s <= log2(N); s++) {
        fftKernel<<<numBlocks, blockSize>>>(d_output, N, s);
    }
    
    cudaMemcpy(output, d_output, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void gpu_fft_cufft(float* input, std::complex<float>* output, int N) {
    cufftHandle plan;
    cufftComplex *d_data;
    
    cudaMalloc(&d_data, N * sizeof(cufftComplex));
    
    std::vector<std::complex<float>> complex_input(N);
    for (int i = 0; i < N; i++) {
        complex_input[i] = std::complex<float>(input[i], 0.0f);
    }
    
    cudaMemcpy(d_data, complex_input.data(), N * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    cudaMemcpy(output, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);
}

bool validate_results(std::complex<float>* result1, std::complex<float>* result2, int N, float tolerance = 1e-2f) {
    bool flag = true;
    for (int i = 0; i < N; i++) {
        float diff_real = std::abs(result1[i].real() - result2[i].real());
        float diff_imag = std::abs(result1[i].imag() - result2[i].imag());
        float magnitude1 = std::sqrt(result1[i].real() * result1[i].real() + 
                                   result1[i].imag() * result1[i].imag());
        float rel_diff = std::sqrt(diff_real * diff_real + diff_imag * diff_imag) / 
                        (magnitude1 + std::numeric_limits<float>::epsilon());
        if (rel_diff > tolerance) {
            std::cout << "Mismatch at index " << i << std::endl;
            std::cout << "Result1: " << result1[i] << std::endl;
            std::cout << "Result2: " << result2[i] << std::endl;
            std::cout << "Relative difference: " << rel_diff << std::endl;
            flag = false;
        }
    }
    return flag;
}

int main() {
    SF_INFO sfinfo;
    SNDFILE* file = sf_open("./image_signal_processing/input.mp3", SFM_READ, &sfinfo);
    if (!file) {
        std::cerr << "Error opening MP3 file" << std::endl;
        return 1;
    }
    
    int N = 8192;
    std::vector<float> input(N);
    sf_read_float(file, input.data(), N);
    sf_close(file);
    
    std::vector<std::complex<float>> cpu_output(N);
    std::vector<std::complex<float>> gpu_output(N);
    std::vector<std::complex<float>> cufft_output(N);
    
    // Timing setup
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    float gpu_time = 0;

    // CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_fft_manual(input.data(), cpu_output.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();

    // GPU timing
    cudaEventRecord(start_gpu);
    gpu_fft_manual(input.data(), gpu_output.data(), N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    
    gpu_fft_cufft(input.data(), cufft_output.data(), N);
    
    std::cout << "Manual CPU vs cuFFT: " 
              << (validate_results(cpu_output.data(), cufft_output.data(), N) ? "MATCH" : "MISMATCH") 
              << std::endl;
    
    std::cout << "Manual GPU vs cuFFT: " 
              << (validate_results(gpu_output.data(), cufft_output.data(), N) ? "MATCH" : "MISMATCH") 
              << std::endl;
    
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // Save results to CSV
    std::ofstream outfile_gpu("./image_signal_processing/fft_gpu_manual.csv");
    std::ofstream outfile_cufft("./image_signal_processing/fft_cufft.csv");
    outfile_gpu << "Index,Frequency,Magnitude,Phase,Real,Imaginary\n";
    outfile_cufft << "Index,Frequency,Magnitude,Phase,Real,Imaginary\n";
    
    float sample_rate = sfinfo.samplerate;
    for(int i = 0; i < N; i++) {
        float freq = i * sample_rate / N;
        
        // Manual GPU implementation results
        float magnitude_gpu = std::abs(gpu_output[i]);
        float phase_gpu = std::arg(gpu_output[i]);
        outfile_gpu << i << "," 
                   << freq << "," 
                   << magnitude_gpu << "," 
                   << phase_gpu << ","
                   << gpu_output[i].real() << ","
                   << gpu_output[i].imag() << "\n";
        
        // cuFFT results
        float magnitude_cufft = std::abs(cufft_output[i]);
        float phase_cufft = std::arg(cufft_output[i]);
        outfile_cufft << i << "," 
                     << freq << "," 
                     << magnitude_cufft << "," 
                     << phase_cufft << ","
                     << cufft_output[i].real() << ","
                     << cufft_output[i].imag() << "\n";
    }
    outfile_gpu.close();
    outfile_cufft.close();

    return 0;
}