#include <torch/extension.h>

torch::Tensor sgm_cuda_forward(torch::Tensor left, torch::Tensor right, uint8_t p1, uint8_t p2);

torch::Tensor sgm_forward(torch::Tensor left, torch::Tensor right, uint8_t p1, uint8_t p2)
{
    return sgm_cuda_forward(left, right, p1, p2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sgm_forward", &sgm_forward, "sgm_forward");
}