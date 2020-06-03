#include "case.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using Eigen::GeneralizedEigenSolver;
using Eigen::MatrixXd;
using Eigen::VectorXcd;

Case1::Case1(int ndim)
    : matM_(MatrixXd::Random(ndim, ndim)), matD_(MatrixXd::Random(ndim, ndim)),
      matK_(MatrixXd::Random(ndim, ndim)), matC_(ndim * 2, ndim * 2),
      matG_(ndim * 2, ndim * 2), ndim_(ndim), alphas(ndim * 2),
      betas(ndim * 2) {
  MatrixXd mzero = MatrixXd::Zero(ndim, ndim);
  MatrixXd mone = MatrixXd::Ones(ndim, ndim);
  matC_ << -matD_, -matK_, mone, mzero;
  matG_ << matM_, mzero, mzero, mone;

  GeneralizedEigenSolver<MatrixXd> ges(matC_, matG_, false);
  alphas = ges.alphas();
  betas = ges.betas();
}
