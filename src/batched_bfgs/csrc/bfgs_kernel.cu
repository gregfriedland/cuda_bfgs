#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

template <typename scalar_t, int dimension>
struct Point {
  scalar_t step;
  scalar_t value;
  scalar_t gradient[dimension];
  scalar_t derivative;
};

template <typename scalar_t, int dimension>
struct LineResult {
  scalar_t step;
  scalar_t value;
  scalar_t gradient[dimension];
  int evaluations;
  bool accepted;
};

struct ExtendedRosenbrock {
  template <typename scalar_t, int dimension>
  __device__ static void evaluate(
      const scalar_t* x, scalar_t& value, scalar_t* gradient) {
    value = scalar_t(0);
#pragma unroll
    for (int index = 0; index < dimension; index += 2) {
      const scalar_t odd = x[index];
      const scalar_t even = x[index + 1];
      const scalar_t residual = even - odd * odd;
      const scalar_t one_minus_odd = scalar_t(1) - odd;
      value += one_minus_odd * one_minus_odd +
          scalar_t(100) * residual * residual;
      gradient[index] = -scalar_t(2) * one_minus_odd -
          scalar_t(400) * odd * residual;
      gradient[index + 1] = scalar_t(200) * residual;
    }
  }
};

struct ExtendedPowell {
  template <typename scalar_t, int dimension>
  __device__ static void evaluate(
      const scalar_t* x, scalar_t& value, scalar_t* gradient) {
    value = scalar_t(0);
#pragma unroll
    for (int index = 0; index < dimension; index += 4) {
      const scalar_t first = x[index] + scalar_t(10) * x[index + 1];
      const scalar_t second = x[index + 2] - x[index + 3];
      const scalar_t third = x[index + 1] - scalar_t(2) * x[index + 2];
      const scalar_t fourth = x[index] - x[index + 3];
      const scalar_t third2 = third * third;
      const scalar_t fourth2 = fourth * fourth;
      value += first * first + scalar_t(5) * second * second +
          third2 * third2 + scalar_t(10) * fourth2 * fourth2;
      gradient[index] = scalar_t(2) * first +
          scalar_t(40) * fourth2 * fourth;
      gradient[index + 1] = scalar_t(20) * first +
          scalar_t(4) * third2 * third;
      gradient[index + 2] = scalar_t(10) * second -
          scalar_t(8) * third2 * third;
      gradient[index + 3] = -scalar_t(10) * second -
          scalar_t(40) * fourth2 * fourth;
    }
  }
};

template <typename scalar_t, int dimension>
__device__ inline scalar_t dot(const scalar_t* first, const scalar_t* second) {
  scalar_t result = scalar_t(0);
#pragma unroll
  for (int index = 0; index < dimension; ++index) {
    result += first[index] * second[index];
  }
  return result;
}

template <typename scalar_t, int dimension>
__device__ inline scalar_t infinity_norm(const scalar_t* values) {
  scalar_t result = scalar_t(0);
#pragma unroll
  for (int index = 0; index < dimension; ++index) {
    result = max(result, fabs(values[index]));
  }
  return result;
}

template <typename scalar_t, int dimension>
__device__ inline scalar_t euclidean_norm(const scalar_t* values) {
  return sqrt(dot<scalar_t, dimension>(values, values));
}

template <typename scalar_t, int dimension>
__device__ inline void copy_vector(const scalar_t* source, scalar_t* target) {
#pragma unroll
  for (int index = 0; index < dimension; ++index) {
    target[index] = source[index];
  }
}

template <typename scalar_t, int dimension, typename Objective>
__device__ inline Point<scalar_t, dimension> evaluate_line(
    const scalar_t* x, const scalar_t* direction, scalar_t step) {
  Point<scalar_t, dimension> point;
  scalar_t trial[dimension];
  point.step = step;
#pragma unroll
  for (int index = 0; index < dimension; ++index) {
    trial[index] = x[index] + step * direction[index];
  }
  Objective::template evaluate<scalar_t, dimension>(
      trial, point.value, point.gradient);
  point.derivative = dot<scalar_t, dimension>(point.gradient, direction);
  return point;
}

template <typename scalar_t, int dimension>
__device__ inline scalar_t cubic_step(
    const Point<scalar_t, dimension>& first,
    const Point<scalar_t, dimension>& second) {
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

template <typename scalar_t, int dimension>
__device__ inline LineResult<scalar_t, dimension> make_line_result(
    const Point<scalar_t, dimension>& point, int evaluations, bool accepted) {
  LineResult<scalar_t, dimension> result;
  result.step = point.step;
  result.value = point.value;
  copy_vector<scalar_t, dimension>(point.gradient, result.gradient);
  result.evaluations = evaluations;
  result.accepted = accepted;
  return result;
}

template <typename scalar_t, int dimension>
__device__ inline LineResult<scalar_t, dimension> failed_line(
    scalar_t value, const scalar_t* gradient, int evaluations) {
  LineResult<scalar_t, dimension> result;
  result.step = scalar_t(0);
  result.value = value;
  copy_vector<scalar_t, dimension>(gradient, result.gradient);
  result.evaluations = evaluations;
  result.accepted = false;
  return result;
}

template <typename scalar_t, int dimension, typename Objective>
__device__ LineResult<scalar_t, dimension> zoom(
    const scalar_t* x,
    const scalar_t* direction,
    scalar_t value0,
    const scalar_t* gradient0,
    scalar_t derivative0,
    scalar_t c1,
    scalar_t c2,
    Point<scalar_t, dimension> low,
    Point<scalar_t, dimension> high,
    int evaluations,
    int max_iterations) {
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const scalar_t step = cubic_step(low, high);
    Point<scalar_t, dimension> trial =
        evaluate_line<scalar_t, dimension, Objective>(x, direction, step);
    ++evaluations;
    const scalar_t armijo = value0 + c1 * step * derivative0;
    const bool bad = !isfinite(trial.value) || trial.value > armijo ||
        trial.value >= low.value;
    if (bad) {
      high = trial;
      continue;
    }
    if (fabs(trial.derivative) <= -c2 * derivative0) {
      return make_line_result(trial, evaluations, true);
    }
    if (trial.derivative * (high.step - low.step) >= scalar_t(0)) {
      high = low;
    }
    low = trial;
  }
  return failed_line<scalar_t, dimension>(value0, gradient0, evaluations);
}

template <typename scalar_t, int dimension, typename Objective>
__device__ LineResult<scalar_t, dimension> strong_wolfe(
    const scalar_t* x,
    const scalar_t* direction,
    scalar_t value0,
    const scalar_t* gradient0,
    scalar_t c1,
    scalar_t c2,
    scalar_t initial_step,
    scalar_t maximum_step,
    int max_bracket_iterations,
    int max_zoom_iterations) {
  const scalar_t derivative0 =
      dot<scalar_t, dimension>(gradient0, direction);
  Point<scalar_t, dimension> previous;
  previous.step = scalar_t(0);
  previous.value = value0;
  copy_vector<scalar_t, dimension>(gradient0, previous.gradient);
  previous.derivative = derivative0;
  scalar_t step = initial_step;
  int evaluations = 0;
  for (int iteration = 0; iteration < max_bracket_iterations; ++iteration) {
    Point<scalar_t, dimension> trial =
        evaluate_line<scalar_t, dimension, Objective>(x, direction, step);
    ++evaluations;
    const scalar_t armijo = value0 + c1 * step * derivative0;
    const bool too_high = !isfinite(trial.value) || trial.value > armijo;
    const bool nondecreasing = iteration > 0 && trial.value >= previous.value;
    if (too_high || nondecreasing) {
      return zoom<scalar_t, dimension, Objective>(
          x, direction, value0, gradient0, derivative0, c1, c2,
          previous, trial, evaluations, max_zoom_iterations);
    }
    if (fabs(trial.derivative) <= -c2 * derivative0) {
      return make_line_result(trial, evaluations, true);
    }
    if (trial.derivative >= scalar_t(0)) {
      return zoom<scalar_t, dimension, Objective>(
          x, direction, value0, gradient0, derivative0, c1, c2,
          trial, previous, evaluations, max_zoom_iterations);
    }
    previous = trial;
    step = min(scalar_t(2) * step, maximum_step);
  }
  return failed_line<scalar_t, dimension>(value0, gradient0, evaluations);
}

template <typename scalar_t, int dimension, typename Objective>
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
  for (std::int64_t member = blockIdx.x * blockDim.x + threadIdx.x;
       member < batch;
       member += blockDim.x * gridDim.x) {
    scalar_t x[dimension];
    scalar_t gradient[dimension];
    scalar_t hessian[dimension * dimension];
    scalar_t direction[dimension];
    scalar_t step_vector[dimension];
    scalar_t change[dimension];
    scalar_t hessian_change[dimension];
#pragma unroll
    for (int row = 0; row < dimension; ++row) {
      x[row] = starts[member * dimension + row];
#pragma unroll
      for (int column = 0; column < dimension; ++column) {
        hessian[row * dimension + column] =
            row == column ? scalar_t(1) : scalar_t(0);
      }
    }
    scalar_t value;
    Objective::template evaluate<scalar_t, dimension>(x, value, gradient);
    int completed_iterations = 0;
    int evaluations = 0;
    bool wolfe_satisfied = true;
    bool converged = infinity_norm<scalar_t, dimension>(gradient) <= tolerance;
    for (int iteration = 0; iteration < max_iterations && !converged;
         ++iteration) {
#pragma unroll
      for (int row = 0; row < dimension; ++row) {
        scalar_t product = scalar_t(0);
#pragma unroll
        for (int column = 0; column < dimension; ++column) {
          product += hessian[row * dimension + column] * gradient[column];
        }
        direction[row] = -product;
      }
      scalar_t derivative = dot<scalar_t, dimension>(gradient, direction);
      if (derivative >= scalar_t(0)) {
#pragma unroll
        for (int row = 0; row < dimension; ++row) {
          direction[row] = -gradient[row];
#pragma unroll
          for (int column = 0; column < dimension; ++column) {
            hessian[row * dimension + column] =
                row == column ? scalar_t(1) : scalar_t(0);
          }
        }
      }
      LineResult<scalar_t, dimension> line =
          strong_wolfe<scalar_t, dimension, Objective>(
              x, direction, value, gradient, c1, c2, initial_step,
              maximum_step, max_bracket_iterations, max_zoom_iterations);
      evaluations += line.evaluations;
      if (!line.accepted) {
        wolfe_satisfied = false;
        break;
      }
#pragma unroll
      for (int index = 0; index < dimension; ++index) {
        step_vector[index] = line.step * direction[index];
        change[index] = line.gradient[index] - gradient[index];
      }
      const scalar_t curvature =
          dot<scalar_t, dimension>(step_vector, change);
      const scalar_t threshold = curvature_epsilon *
          euclidean_norm<scalar_t, dimension>(step_vector) *
          euclidean_norm<scalar_t, dimension>(change);
      if (isfinite(curvature) && curvature > threshold) {
#pragma unroll
        for (int row = 0; row < dimension; ++row) {
          scalar_t product = scalar_t(0);
#pragma unroll
          for (int column = 0; column < dimension; ++column) {
            product += hessian[row * dimension + column] * change[column];
          }
          hessian_change[row] = product;
        }
        const scalar_t y_h_y =
            dot<scalar_t, dimension>(change, hessian_change);
        const scalar_t coefficient = (curvature + y_h_y) /
            (curvature * curvature);
#pragma unroll
        for (int row = 0; row < dimension; ++row) {
#pragma unroll
          for (int column = 0; column < dimension; ++column) {
            hessian[row * dimension + column] +=
                coefficient * step_vector[row] * step_vector[column] -
                (hessian_change[row] * step_vector[column] +
                 step_vector[row] * hessian_change[column]) /
                    curvature;
          }
        }
      }
#pragma unroll
      for (int index = 0; index < dimension; ++index) {
        x[index] += step_vector[index];
        gradient[index] = line.gradient[index];
      }
      value = line.value;
      ++completed_iterations;
      converged =
          infinity_norm<scalar_t, dimension>(gradient) <= tolerance;
      const bool stagnant =
          infinity_norm<scalar_t, dimension>(step_vector) <= step_tolerance;
      if (stagnant && !converged) {
        break;
      }
    }
#pragma unroll
    for (int index = 0; index < dimension; ++index) {
      output_x[member * dimension + index] = x[index];
      output_gradient[member * dimension + index] = gradient[index];
    }
    output_value[member] = value;
    output_iterations[member] = completed_iterations;
    output_evaluations[member] = evaluations;
    output_converged[member] = converged;
    output_wolfe[member] = wolfe_satisfied;
  }
}

template <typename scalar_t, int dimension, typename Objective>
void launch_kernel(
    const torch::Tensor& starts,
    torch::Tensor& output_x,
    torch::Tensor& output_value,
    torch::Tensor& output_gradient,
    torch::Tensor& output_iterations,
    torch::Tensor& output_evaluations,
    torch::Tensor& output_converged,
    torch::Tensor& output_wolfe,
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
  const std::int64_t batch = starts.size(0);
  constexpr int threads = 128;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (batch + threads - 1) / threads, 4096));
  bfgs_kernel<scalar_t, dimension, Objective><<<
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

}  // namespace

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
  AT_DISPATCH_FLOATING_TYPES(starts.scalar_type(), "bfgs_cuda", [&] {
    const auto scalar_c1 = static_cast<scalar_t>(c1);
    const auto scalar_c2 = static_cast<scalar_t>(c2);
    const auto scalar_tolerance = static_cast<scalar_t>(tolerance);
    const auto scalar_step_tolerance = static_cast<scalar_t>(step_tolerance);
    const auto scalar_curvature_epsilon =
        static_cast<scalar_t>(curvature_epsilon);
    const auto scalar_initial_step = static_cast<scalar_t>(initial_step);
    const auto scalar_maximum_step = static_cast<scalar_t>(maximum_step);
#define LAUNCH(dimension, objective_type)                                      \
    launch_kernel<scalar_t, dimension, objective_type>(                        \
        starts, output_x, output_value, output_gradient, output_iterations,    \
        output_evaluations, output_converged, output_wolfe, scalar_c1,         \
        scalar_c2, scalar_tolerance, scalar_step_tolerance,                    \
        scalar_curvature_epsilon, scalar_initial_step, scalar_maximum_step,    \
        static_cast<int>(max_iterations),                                      \
        static_cast<int>(max_bracket_iterations),                              \
        static_cast<int>(max_zoom_iterations))
    if (starts.size(1) == 2) {
      LAUNCH(2, ExtendedRosenbrock);
    } else if (objective == 0) {
      LAUNCH(16, ExtendedRosenbrock);
    } else {
      LAUNCH(16, ExtendedPowell);
    }
#undef LAUNCH
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
