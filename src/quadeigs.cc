#include "quadeigs.hpp"
#include "soar.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <fmt/format.h>

using namespace Eigen;

QuadEigs::QuadEigs(const Ref<const MatrixXd> &matM,
                   const Ref<const MatrixXd> &matD,
                   const Ref<const MatrixXd> &matK)
    : ndim_(matM.cols()), matM_(matM), matD_(matD), matK_(matK),
      matA_(ndim_, ndim_), matB_(ndim_, ndim_) {
  MatrixXd matMi = matM.inverse();
  matA_ = -matMi * matD_;
  matB_ = -matMi * matK_;
}

VectorXcd QuadEigs::eigenvalues(int m) {
  Soar soar(matA_, matB_);
  MatrixXd matQm = soar.compute(m);
  MatrixXd matMm = matQm.transpose() * matM_ * matQm;
  MatrixXd matDm = matQm.transpose() * matD_ * matQm;
  MatrixXd matKm = matQm.transpose() * matK_ * matQm;

  MatrixXd matC(2 * m, 2 * m);
  MatrixXd matG(2 * m, 2 * m);
  MatrixXd mzero = MatrixXd::Zero(m, m);
  MatrixXd mone = MatrixXd::Identity(m, m);
  matC << -matDm, -matKm, mone, mzero;
  matG << matMm, mzero, mzero, mone;

  GeneralizedEigenSolver<MatrixXd> ges(matC, matG, false);
  VectorXcd alphas = ges.alphas();
  VectorXd betas = ges.betas();

  fmt::print("{:>30s}, {:>15s}\n", "alphas", "betas");
  for (int i = 0; i < m; ++i) {
    if (std::abs(betas(i)) > 1.0e-8) {
      fmt::print("{:14.5f}+{:14.5f}i\n", alphas(i).real() / betas(i),
                 alphas(i).imag() / betas(i));
    }
  }

  VectorXcd ret(m);
  return ret;
}