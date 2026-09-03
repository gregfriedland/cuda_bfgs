# Batched BFGS algorithm contract

All three implementations optimize independent two-dimensional Rosenbrock
functions:

\[
f(x, y) = (1 - x)^2 + 100(y - x^2)^2.
\]

The implementations share this contract:

- Initialize the inverse-Hessian approximation to the identity matrix.
- Use `-H g` as the search direction. Reset `H` to identity and use `-g` if
  numerical error makes the direction non-descending.
- Use a strong-Wolfe line search with `c1=1e-4`, `c2=0.9`, initial step 1,
  maximum step 64, 20 bracketing iterations, and 25 zoom iterations.
- Use safeguarded cubic interpolation. Fall back to the bracket midpoint when
  the cubic proposal is non-finite or falls within 10% of either endpoint.
- Treat a non-finite trial objective as an upper-bracket observation.
- Skip the inverse-BFGS update when
  `s·y <= curvature_eps * ||s|| * ||y||`.
- Converge when the infinity norm of the gradient is at most `tolerance`.
- Stop without convergence if the line search fails or the accepted step has
  infinity norm at most `step_tolerance` while the gradient remains large.

The CUDA implementation intentionally compiles the Rosenbrock objective into
the kernel. A CUDA kernel cannot call an arbitrary Python objective callback.
One CUDA thread owns one complete 2D optimization, including its line search
and 2x2 inverse Hessian.
