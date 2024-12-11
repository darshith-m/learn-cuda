#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <iostream>

int main() {
    // Host vector
    thrust::host_vector<int> h_vec{1, 2, 3, 4, 5};

    // Transfer to device
    thrust::device_vector<int> d_vec = h_vec;

    // Perform inclusive scan
    thrust::inclusive_scan(
        d_vec.begin(),     // Input start
        d_vec.end(),       // Input end
        d_vec.begin()      // Output start (in-place)
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
