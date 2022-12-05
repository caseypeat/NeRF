#include <iostream>
#include <vector>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void fill_kernel(
    scalar_t* __restrict__ tensor,
    bool* __restrict__ mask,
    int n_samples)
{
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * n_samples;
    // int stride = blockDim.x * gridDim.x
    // int stride = 1;
    for (int i = index+1; i < index+n_samples; i++) {
        // tensor[i] = tensor[i-1];
        if (mask[i] == 0) {
            tensor[i] = tensor[i-1];
        }
    }
}

void fill_cuda(torch::Tensor tensor, torch::Tensor mask, int n_samples)
{
    const int blocks = 16;
    const int threads = 1024;
    AT_DISPATCH_FLOATING_TYPES(tensor.type(), "fill_cuda", ([&] {
        fill_kernel<<<blocks, threads>>>(
            tensor.data<scalar_t>(),
            mask.data<bool>(),
            n_samples
        );
    }));
}