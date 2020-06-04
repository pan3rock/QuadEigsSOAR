#include "case.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;

Case1::Case1(const Ref<const MatrixXd> &matM, const Ref<const MatrixXd> &matD,
             const Ref<const MatrixXd> &matK)

    : ndim_(matM.cols()), matM_(matM), matD_(matD), matK_(matK),
      matC_(ndim_ * 2, ndim_ * 2), matG_(ndim_ * 2, ndim_ * 2),
      alphas(ndim_ * 2), betas(ndim_ * 2) {
  MatrixXd mzero = MatrixXd::Zero(ndim_, ndim_);
  MatrixXd mone = MatrixXd::Identity(ndim_, ndim_);
  matC_ << -matD_, -matK_, mone, mzero;
  matG_ << matM_, mzero, mzero, mone;

  GeneralizedEigenSolver<MatrixXd> ges(matC_, matG_, false);
  alphas = ges.alphas();
  betas = ges.betas();
}
