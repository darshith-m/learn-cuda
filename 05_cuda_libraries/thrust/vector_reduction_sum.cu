#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>

int main() {
    // Host vector
    thrust::host_vector<int> h_vec{1, 2, 3, 4, 5};

    // Transfer to device
    thrust::device_vector<int> d_vec = h_vec;

    // Compute sum
    // Sum all elements in device vector
    int sum = thrust::reduce(
        d_vec.begin(),         // Start of input range
        d_vec.end(),          // End of input range
        0,                    // Initial value for reduction
        thrust::plus<int>()   // Binary operation for reduction
    );

    // Print result
    std::cout << "Sum: " << sum << std::endl;

    return 0;
}
