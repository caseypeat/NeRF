#include <iostream>

#include <torch/extension.h>

#include <pybind11/pybind11.h>

// #include <fill_kernel.cu>


// using namespace torch::indexing;

// void fill(torch::Tensor tensor, torch::Tensor mask_inv, int n_samples) {
//     for (int i=1; i<n_samples; i++) {
//         auto tensor_slice = tensor.index({"...", i});
//         auto tensor_slice_ = tensor.index({"...", i-1});
//         auto mask_inv_slice = mask_inv.index({"...", i});

//         tensor_slice.index_put_({mask_inv_slice}, tensor_slice_.index({mask_inv_slice}));
//     }
// }

void fill_cuda(torch::Tensor tensor, torch::Tensor mask, int n_samples);

void fill(torch::Tensor tensor, torch::Tensor mask, int n_samples) {
    // const int blocks = 16;
    // const int threads = 1024;
    // fill_kernel<scalar_t><<<blocks, threads>>>(
    //     gates.data<scalar_t>(),
    // );
    fill_cuda(tensor, mask, n_samples);
}

PYBIND11_MODULE(fill, m) {
    m.def("fill", &fill, "fill function");
}