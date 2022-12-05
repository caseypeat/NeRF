#include <iostream>
#include <vector>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void fill_cuda_kernel(
    scalar_t* __restrict__ tensor,
    bool* __restrict__ mask,
    int n_samples)
{
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * n_samples;

    for (int i = index+1; i < index+n_samples; i++) {
        // tensor[i] = tensor[i-1];
        if (mask[i] == 0) {
            tensor[i] = tensor[i-1];
        }
    }
}

void fill_cuda(torch::Tensor tensor, torch::Tensor mask, int n_rays, int n_samples)
{   
    const int threads = 1024;
    const int blocks = n_rays / threads;

    AT_DISPATCH_FLOATING_TYPES(tensor.type(), "fill_cuda", ([&] {
        fill_cuda_kernel<<<blocks, threads>>>(
            tensor.data<scalar_t>(),
            mask.data<bool>(),
            n_samples
        );
    }));
}


template <typename scalar_t>
__global__ void fill_cuda_acc_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tensor,
    torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> mask,
    int n_samples)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 1; i < n_samples; i++) {
        if (mask[index][i] == 0) {
            tensor[index][i] = tensor[index][i-1];
        }
    }
}

void fill_cuda_acc(torch::Tensor tensor, torch::Tensor mask, int n_rays, int n_samples)
{   
    const int threads = 1024;
    const int blocks = n_rays / threads;

    AT_DISPATCH_FLOATING_TYPES(tensor.type(), "fill_cuda", ([&] {
        fill_cuda_acc_kernel<<<blocks, threads>>>(
            tensor.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
            n_samples
        );
    }));
}