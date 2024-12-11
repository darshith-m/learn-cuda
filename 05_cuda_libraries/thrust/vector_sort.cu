#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>

int main() {
    // Host vector
    thrust::host_vector<int> h_vec{5, 1, 4, 2, 3};

    // Transfer to device
    thrust::device_vector<int> d_vec = h_vec;

    // Sort on device
    thrust::sort(
        d_vec.begin(),     // Start of device vector
        d_vec.end()        // End of device vector
    );

    // Copy back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    // Print result
    for (int val : h_vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
