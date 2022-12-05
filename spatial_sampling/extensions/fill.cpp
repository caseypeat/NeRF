#include <iostream>

#include <torch/extension.h>

#include <pybind11/pybind11.h>


void fill_cuda_acc(torch::Tensor tensor, torch::Tensor mask, int n_rays, int n_samples);

void fill(torch::Tensor tensor, torch::Tensor mask, int n_rays, int n_samples) {
    fill_cuda_acc(tensor, mask, n_rays, n_samples);
}

PYBIND11_MODULE(fill, m) {
    m.def("fill", &fill, "fill function");
}