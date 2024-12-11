#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <iostream>

// Functor to check if a number is even
struct is_even {
    __host__ __device__
    bool operator()(const int& x) const {
        return x % 2 == 0;
    }
};

int main() {
    // Host vector
    thrust::host_vector<int> h_vec{1, 2, 3, 4, 5, 6, 7, 8};

    // Transfer to device
    thrust::device_vector<int> d_vec = h_vec;

    // Device vector for results
    thrust::device_vector<int> d_result(d_vec.size());

    // Copy even numbers to d_result
    auto end = thrust::copy_if(
        d_vec.begin(),         // Input start
        d_vec.end(),          // Input end
        d_result.begin(),     // Output start
        is_even()             // Predicate functor
    );

    // Resize result vector to actual size
    d_result.resize(end - d_result.begin());

    // Copy back to host
    thrust::host_vector<int> h_result = d_result;

    // Print result
    for (int val : h_result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
