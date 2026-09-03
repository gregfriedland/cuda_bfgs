#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

template <typename scalar_t>
struct Point {
  scalar_t step;
  scalar_t value;
  scalar_t gradient0;
  scalar_t gradient1;
  scalar_t derivative;
};

template <typename scalar_t>
struct LineResult {
  scalar_t step;
  scalar_t value;
  scalar_t gradient0;
  scalar_t gradient1;
  int evaluations;
  bool accepted;
};

template <typename scalar_t>
__device__ inline void rosenbrock(
    scalar_t x0,
    scalar_t x1,
    scalar_t& value,
    scalar_t& gradient0,
    scalar_t& gradient1) {
  const scalar_t residual = x1 - x0 * x0;
  const scalar_t one_minus_x = scalar_t(1) - x0;
  value = one_minus_x * one_minus_x + scalar_t(100) * residual * residual;
  gradient0 = -scalar_t(2) * one_minus_x -
      scalar_t(400) * x0 * residual;
  gradient1 = scalar_t(200) * residual;
}

template <typename scalar_t>
__device__ inline Point<scalar_t> evaluate(
    scalar_t x0,
    scalar_t x1,
    scalar_t direction0,
    scalar_t direction1,
    scalar_t step) {
  Point<scalar_t> point;
  point.step = step;
  rosenbrock(
      x0 + step * direction0,
      x1 + step * direction1,
      point.value,
      point.gradient0,
      point.gradient1);
  point.derivative = point.gradient0 * direction0 +
      point.gradient1 * direction1;
  return point;
}

template <typename scalar_t>
__device__ inline scalar_t cubic_step(
    const Point<scalar_t>& first,
    const Point<scalar_t>& second) {
  const scalar_t lower = min(first.step, second.step);
  const scalar_t upper = max(first.step, second.step);
  const scalar_t midpoint = scalar_t(0.5) * (lower + upper);
  const scalar_t separation = first.step - second.step;
  if (separation == scalar_t(0)) {
    return midpoint;
  }
  const scalar_t d1 = first.derivative + second.derivative -
      scalar_t(3) * (first.value - second.value) / separation;
  const scalar_t discriminant = d1 * d1 -
      first.derivative * second.derivative;
  if (!(discriminant >= scalar_t(0)) || !isfinite(discriminant)) {
    return midpoint;
  }
  const scalar_t d2 = sqrt(discriminant);
  scalar_t denominator;
  scalar_t candidate;
  if (first.step <= second.step) {
    denominator = second.derivative - first.derivative + scalar_t(2) * d2;
    candidate = second.step - (second.step - first.step) *
        (second.derivative + d2 - d1) / denominator;
  } else {
    denominator = first.derivative - second.derivative + scalar_t(2) * d2;
    candidate = first.step - (first.step - second.step) *
        (first.derivative + d2 - d1) / denominator;
  }
  const scalar_t guard = scalar_t(0.1) * (upper - lower);
  const bool usable = isfinite(candidate) && isfinite(denominator) &&
      fabs(denominator) > scalar_t(1e-20) &&
      candidate > lower + guard && candidate < upper - guard;
  return usable ? candidate : midpoint;
}

template <typename scalar_t>
__device__ LineResult<scalar_t> zoom(
    scalar_t x0,
    scalar_t x1,
    scalar_t direction0,
    scalar_t direction1,
    scalar_t value0,
    scalar_t derivative0,
    scalar_t c1,
    scalar_t c2,
    Point<scalar_t> low,
    Point<scalar_t> high,
    int evaluations,
    int max_iterations) {
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const scalar_t step = cubic_step(low, high);
    Point<scalar_t> trial = evaluate(
        x0, x1, direction0, direction1, step);
    ++evaluations;
    const scalar_t armijo = value0 + c1 * step * derivative0;
    const bool bad = !isfinite(trial.value) || trial.value > armijo ||
        trial.value >= low.value;
    if (bad) {
      high = trial;
      continue;
    }
    if (fabs(trial.derivative) <= -c2 * derivative0) {
      return {
          trial.step,
          trial.value,
          trial.gradient0,
          trial.gradient1,
          evaluations,
          true};
    }
    if (trial.derivative * (high.step - low.step) >= scalar_t(0)) {
      high = low;
    }
    low = trial;
  }
  return {scalar_t(0), value0, scalar_t(0), scalar_t(0), evaluations, false};
}

template <typename scalar_t>
__device__ LineResult<scalar_t> strong_wolfe(
    scalar_t x0,
    scalar_t x1,
    scalar_t direction0,
    scalar_t direction1,
    scalar_t value0,
    scalar_t gradient0,
    scalar_t gradient1,
    scalar_t c1,
    scalar_t c2,
    scalar_t initial_step,
    scalar_t maximum_step,
    int max_bracket_iterations,
    int max_zoom_iterations) {
  const scalar_t derivative0 = gradient0 * direction0 +
      gradient1 * direction1;
  Point<scalar_t> previous = {
      scalar_t(0), value0, gradient0, gradient1, derivative0};
  scalar_t step = initial_step;
  int evaluations = 0;
  for (int iteration = 0; iteration < max_bracket_iterations; ++iteration) {
    Point<scalar_t> trial = evaluate(
        x0, x1, direction0, direction1, step);
    ++evaluations;
    const scalar_t armijo = value0 + c1 * step * derivative0;
    const bool too_high = !isfinite(trial.value) || trial.value > armijo;
    const bool nondecreasing = iteration > 0 && trial.value >= previous.value;
    if (too_high || nondecreasing) {
      return zoom(
          x0, x1, direction0, direction1, value0, derivative0, c1, c2,
          previous, trial, evaluations, max_zoom_iterations);
    }
    if (fabs(trial.derivative) <= -c2 * derivative0) {
      return {
          trial.step,
          trial.value,
          trial.gradient0,
          trial.gradient1,
          evaluations,
          true};
    }
    if (trial.derivative >= scalar_t(0)) {
      return zoom(
          x0, x1, direction0, direction1, value0, derivative0, c1, c2,
          trial, previous, evaluations, max_zoom_iterations);
    }
    previous = trial;
    step = min(scalar_t(2) * step, maximum_step);
  }
  return {scalar_t(0), value0, gradient0, gradient1, evaluations, false};
}

template <typename scalar_t>
__global__ void bfgs_kernel(
    const scalar_t* starts,
    scalar_t* output_x,
    scalar_t* output_value,
    scalar_t* output_gradient,
    std::int32_t* output_iterations,
    std::int32_t* output_evaluations,
    bool* output_converged,
    bool* output_wolfe,
    std::int64_t batch,
    scalar_t c1,
    scalar_t c2,
    scalar_t tolerance,
    scalar_t step_tolerance,
    scalar_t curvature_epsilon,
    scalar_t initial_step,
    scalar_t maximum_step,
    int max_iterations,
    int max_bracket_iterations,
    int max_zoom_iterations) {
  for (std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < batch;
       index += blockDim.x * gridDim.x) {
    scalar_t x0 = starts[2 * index];
    scalar_t x1 = starts[2 * index + 1];
    scalar_t h00 = scalar_t(1);
    scalar_t h01 = scalar_t(0);
    scalar_t h11 = scalar_t(1);
    scalar_t value;
    scalar_t gradient0;
    scalar_t gradient1;
    rosenbrock(x0, x1, value, gradient0, gradient1);
    int completed_iterations = 0;
    int evaluations = 0;
    bool wolfe_satisfied = true;
    bool converged = max(fabs(gradient0), fabs(gradient1)) <= tolerance;
    for (int iteration = 0; iteration < max_iterations && !converged;
         ++iteration) {
      scalar_t direction0 = -(h00 * gradient0 + h01 * gradient1);
      scalar_t direction1 = -(h01 * gradient0 + h11 * gradient1);
      scalar_t derivative = gradient0 * direction0 +
          gradient1 * direction1;
      if (derivative >= scalar_t(0)) {
        h00 = scalar_t(1);
        h01 = scalar_t(0);
        h11 = scalar_t(1);
        direction0 = -gradient0;
        direction1 = -gradient1;
      }
      LineResult<scalar_t> line = strong_wolfe(
          x0, x1, direction0, direction1, value, gradient0, gradient1,
          c1, c2, initial_step, maximum_step, max_bracket_iterations,
          max_zoom_iterations);
      evaluations += line.evaluations;
      if (!line.accepted) {
        wolfe_satisfied = false;
        break;
      }
      const scalar_t step0 = line.step * direction0;
      const scalar_t step1 = line.step * direction1;
      const scalar_t change0 = line.gradient0 - gradient0;
      const scalar_t change1 = line.gradient1 - gradient1;
      const scalar_t curvature = step0 * change0 + step1 * change1;
      const scalar_t threshold = curvature_epsilon *
          hypot(step0, step1) * hypot(change0, change1);
      if (isfinite(curvature) && curvature > threshold) {
        const scalar_t hy0 = h00 * change0 + h01 * change1;
        const scalar_t hy1 = h01 * change0 + h11 * change1;
        const scalar_t yhy = change0 * hy0 + change1 * hy1;
        const scalar_t coefficient = (curvature + yhy) /
            (curvature * curvature);
        h00 += coefficient * step0 * step0 -
            scalar_t(2) * hy0 * step0 / curvature;
        h01 += coefficient * step0 * step1 -
            (hy0 * step1 + step0 * hy1) / curvature;
        h11 += coefficient * step1 * step1 -
            scalar_t(2) * hy1 * step1 / curvature;
      }
      x0 += step0;
      x1 += step1;
      value = line.value;
      gradient0 = line.gradient0;
      gradient1 = line.gradient1;
      ++completed_iterations;
      converged = max(fabs(gradient0), fabs(gradient1)) <= tolerance;
      const bool stagnant = max(fabs(step0), fabs(step1)) <= step_tolerance;
      if (stagnant && !converged) {
        break;
      }
    }
    output_x[2 * index] = x0;
    output_x[2 * index + 1] = x1;
    output_value[index] = value;
    output_gradient[2 * index] = gradient0;
    output_gradient[2 * index + 1] = gradient1;
    output_iterations[index] = completed_iterations;
    output_evaluations[index] = evaluations;
    output_converged[index] = converged;
    output_wolfe[index] = wolfe_satisfied;
  }
}

}  // namespace

std::vector<torch::Tensor> optimize_cuda(
    torch::Tensor starts,
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
  const std::int64_t batch = starts.size(0);
  torch::Tensor output_x = torch::empty_like(starts);
  torch::Tensor output_value = torch::empty({batch}, starts.options());
  torch::Tensor output_gradient = torch::empty_like(starts);
  const auto int_options = starts.options().dtype(torch::kInt32);
  const auto bool_options = starts.options().dtype(torch::kBool);
  torch::Tensor output_iterations = torch::empty({batch}, int_options);
  torch::Tensor output_evaluations = torch::empty({batch}, int_options);
  torch::Tensor output_converged = torch::empty({batch}, bool_options);
  torch::Tensor output_wolfe = torch::empty({batch}, bool_options);
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (batch + threads - 1) / threads, 4096));
  AT_DISPATCH_FLOATING_TYPES(starts.scalar_type(), "bfgs_cuda", [&] {
    bfgs_kernel<scalar_t><<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        starts.data_ptr<scalar_t>(),
        output_x.data_ptr<scalar_t>(),
        output_value.data_ptr<scalar_t>(),
        output_gradient.data_ptr<scalar_t>(),
        output_iterations.data_ptr<std::int32_t>(),
        output_evaluations.data_ptr<std::int32_t>(),
        output_converged.data_ptr<bool>(),
        output_wolfe.data_ptr<bool>(),
        batch,
        static_cast<scalar_t>(c1),
        static_cast<scalar_t>(c2),
        static_cast<scalar_t>(tolerance),
        static_cast<scalar_t>(step_tolerance),
        static_cast<scalar_t>(curvature_epsilon),
        static_cast<scalar_t>(initial_step),
        static_cast<scalar_t>(maximum_step),
        static_cast<int>(max_iterations),
        static_cast<int>(max_bracket_iterations),
        static_cast<int>(max_zoom_iterations));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      output_x,
      output_value,
      output_gradient,
      output_iterations,
      output_evaluations,
      output_converged,
      output_wolfe};
}
