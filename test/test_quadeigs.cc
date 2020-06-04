#include "case.hpp"
#include "catch.hpp"
#include "quadeigs.hpp"
#include "timer.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <fmt/format.h>
#include <iostream>
#include <vector>

using namespace Eigen;
using Td = Triplet<double>;

TEST_CASE("random (4, 4)", "[quadeigs]") {
  int ndim = 4;
  MatrixXd matM = MatrixXd::Random(ndim, ndim);
  MatrixXd matD = MatrixXd::Random(ndim, ndim);
  MatrixXd matK = MatrixXd::Random(ndim, ndim);
  Case1 c1(matM, matD, matK);
  fmt::print("{:>30s}, {:>15s}\n", "alphas", "betas");
  for (int i = 0; i < ndim; ++i) {
    fmt::print("{:14.5f}+{:14.5f}i, {:15.5f}\n", c1.alphas(i).real(),
               c1.alphas(i).imag(), c1.betas(i));
  }

  QuadEigs qe(matM, matD, matK);
  int m = 4;
  VectorXcd ev = qe.eigenvalues(m);
}

TEST_CASE("random (10, 10)", "[quadeigs]") {
  int ndim = 10;
  MatrixXd matM = MatrixXd::Random(ndim, ndim);
  MatrixXd matD = MatrixXd::Random(ndim, ndim);
  MatrixXd matK = MatrixXd::Random(ndim, ndim);
  Case1 c1(matM, matD, matK);
  for (int i = 0; i < ndim; ++i) {
    if (std::abs(c1.betas(i)) > 1.0e-8) {
      fmt::print("{:14.5f}+{:14.5f}i\n", c1.alphas(i).real() / c1.betas(i),
                 c1.alphas(i).imag() / c1.betas(i));
    }
  }
  fmt::print("---------------------------------------------------\n");

  QuadEigs qe(matM, matD, matK);
  int m = 4;
  VectorXcd ev = qe.eigenvalues(m);
}

TEST_CASE("Identity (20, 20)", "[quadeigs]") {
  int ndim = 20;
  MatrixXd matM = MatrixXd::Identity(ndim, ndim);
  MatrixXd matD = MatrixXd::Identity(ndim, ndim);
  MatrixXd matK = MatrixXd::Identity(ndim, ndim) * 0.2;
  for (int i = 1; i < ndim; ++i) {
    matK(i, i - 1) = -0.1;
    matK(i - 1, i) = -0.1;
  }
  std::cout << matK.inverse() << std::endl;
  Case1 c1(matM, matD, matK);
  for (int i = 0; i < ndim; ++i) {
    if (std::abs(c1.betas(i)) > 1.0e-8) {
      fmt::print("{:14.5f}+{:14.5f}i\n", c1.alphas(i).real() / c1.betas(i),
                 c1.alphas(i).imag() / c1.betas(i));
    }
  }
  fmt::print("---------------------------------------------------\n");

  QuadEigs qe(matM, matD, matK);
  int m = 4;
  VectorXcd ev = qe.eigenvalues(m);
}

TEST_CASE("inverse benchmark", "[quadeigs]") {
  int ndim = 20;
  MatrixXd matD = MatrixXd::Identity(ndim, ndim) * 2;
  for (int i = 1; i < ndim; ++i) {
    matD(i, i - 1) = -1.0;
    matD(i - 1, i) = -1.0;
  }
  MatrixXd matId = MatrixXd::Identity(ndim, ndim);
  Timer::begin("inv-dense");
  MatrixXd matDi = matD.ldlt().solve(matId);
  Timer::end("inv-dense");

  SparseMatrix<double> matS(ndim, ndim);
  std::vector<Td> tl;
  for (int i = 0; i < ndim; ++i) {
    tl.push_back(Td(i, i, 2.0));
  }
  for (int i = 1; i < ndim; ++i) {
    tl.push_back(Td(i, i - 1, -1.0));
    tl.push_back(Td(i - 1, i, -1.0));
  }
  matS.setFromTriplets(tl.begin(), tl.end());
  SparseMatrix<double> matI(ndim, ndim);
  matI.setIdentity();

  Timer::begin("inv-sparse");
  SimplicialLDLT<SparseMatrix<double>> solver;
  solver.compute(matS);
  SparseMatrix<double> matSi = solver.solve(matI);
  Timer::end("inv-sparse");

  std::cout << Timer::summery() << std::endl;
}