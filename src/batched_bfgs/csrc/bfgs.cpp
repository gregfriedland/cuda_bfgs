#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> optimize_cuda(
    torch::Tensor starts,
    std::int64_t objective,
    double c1,
    double c2,
    double tolerance,
    double step_tolerance,
    double curvature_epsilon,
    double initial_step,
    double maximum_step,
    std::int64_t max_iterations,
    std::int64_t max_bracket_iterations,
    std::int64_t max_zoom_iterations);

std::vector<torch::Tensor> optimize(
    torch::Tensor starts,
    std::int64_t objective,
    double c1,
    double c2,
    double tolerance,
    double step_tolerance,
    double curvature_epsilon,
    double initial_step,
    double maximum_step,
    std::int64_t max_iterations,
    std::int64_t max_bracket_iterations,
    std::int64_t max_zoom_iterations) {
  TORCH_CHECK(starts.is_cuda(), "starts must be a CUDA tensor");
  TORCH_CHECK(starts.is_contiguous(), "starts must be contiguous");
  TORCH_CHECK(starts.dim() == 2, "starts must be rank two");
  TORCH_CHECK(
      starts.size(1) == 2 || starts.size(1) == 16,
      "starts must have shape [batch, 2] or [batch, 16]");
  TORCH_CHECK(objective == 0 || objective == 1, "unknown objective");
  TORCH_CHECK(
      starts.size(1) == 16 || objective == 0,
      "2D CUDA optimization supports only Rosenbrock");
  TORCH_CHECK(starts.size(0) > 0, "starts must contain a batch member");
  TORCH_CHECK(
      starts.scalar_type() == torch::kFloat32 ||
          starts.scalar_type() == torch::kFloat64,
      "starts must use float32 or float64");
  return optimize_cuda(
      starts,
      objective,
      c1,
      c2,
      tolerance,
      step_tolerance,
      curvature_epsilon,
      initial_step,
      maximum_step,
      max_iterations,
      max_bracket_iterations,
      max_zoom_iterations);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("optimize", &optimize, "Batched strong-Wolfe BFGS (CUDA)");
}
