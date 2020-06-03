#include "case.hpp"
#include "catch.hpp"

#include <Eigen/Dense>
#include <fmt/format.h>
#include <iostream>

using Eigen::IOFormat;

TEST_CASE("random (4, 4)", "[quadeigs]") {
  int ndim = 4;
  Case1 c1(ndim);
  fmt::print("{:>30s}, {:>15s}\n", "alphas", "betas");
  for (int i = 0; i < ndim; ++i) {
    fmt::print("{:14.5f}+{:14.5f}i, {:15.5f}\n", c1.alphas(i).real(),
               c1.alphas(i).imag(), c1.betas(i));
  }
}