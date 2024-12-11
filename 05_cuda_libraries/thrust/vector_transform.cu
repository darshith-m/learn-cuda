#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>

// Custom square functor
struct square {
    __host__ __device__
    float operator()(const float& x) const {
        return x * x;
    }
};

int main() {
    // Host vector
    thrust::host_vector<float> h_vec(5);
    h_vec[0] = 1.0f; h_vec[1] = 2.0f; h_vec[2] = 3.0f; h_vec[3] = 4.0f; h_vec[4] = 5.0f;

    // Transfer to device
    thrust::device_vector<float> d_vec = h_vec;

    // Apply square transformation
    thrust::transform(
        d_vec.begin(),     // Input iterator - start of input range
        d_vec.end(),       // Input iterator - end of input range
        d_vec.begin(),     // Output iterator - where to write results
        square()           // Function object instance to apply to each element
    );

    // Copy back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    // Print result
    for (float val : h_vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
