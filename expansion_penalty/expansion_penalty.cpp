#include <torch/extension.h>
#include <vector>

int expansion_penalty_cuda_forward(at::Tensor xyz, int primitive_size, at::Tensor father, at::Tensor dist, double alpha, at::Tensor neighbor, at::Tensor cost, at::Tensor mean_mst_length);

int expansion_penalty_cuda_backward(at::Tensor xyz, at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx);

int expansion_penalty_forward(at::Tensor xyz, int primitive_size, at::Tensor father, at::Tensor dist, double alpha, at::Tensor neighbor, at::Tensor cost, at::Tensor mean_mst_length) {
	return expansion_penalty_cuda_forward(xyz, primitive_size, father, dist, alpha, neighbor, cost, mean_mst_length);
}

int expansion_penalty_backward(at::Tensor xyz, at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx) {

    return expansion_penalty_cuda_backward(xyz, gradxyz, graddist, idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &expansion_penalty_forward, "expansion_penalty forward (CUDA)");
  m.def("backward", &expansion_penalty_backward, "expansion_penalty backward (CUDA)");
}