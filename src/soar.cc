#include "soar.hpp"

#include <Eigen/Dense>
#include <fmt/format.h>
#include <vector>

using namespace Eigen;

Soar::Soar(const Ref<const MatrixXd> &matA, const Ref<const MatrixXd> &matB)
    : ndim_(matA.rows()), matA_(matA), matB_(matB),
      u_(VectorXd::Random(ndim_)) {}

MatrixXd Soar::compute(int n) {
  VectorXd q = u_ / u_.norm();
  VectorXd f = VectorXd::Zero(ndim_);

  // initialize
  MatrixXd matQ = MatrixXd::Zero(ndim_, n);
  MatrixXd matP = MatrixXd::Zero(ndim_, n);
  MatrixXd matT = MatrixXd::Zero(n, n);
  std::vector<int> deflation;

  matQ.col(0) = q;

  for (int i = 0; i < n - 1; ++i) {
    // Recurrence role
    VectorXd r = matA_ * matQ.col(i) + matB_ * matP.col(i);
    double norm_init = r.norm();
    MatrixXd basis = matQ.leftCols(i + 1);

    // Modified Gram Schmidt procedure
    // First orthogonalization
    VectorXd coef = VectorXd::Zero(i + 1);
    for (int j = 0; j < i + 1; ++j) {
      // Projection coeficients and projection subtraction
      VectorXd v = basis.col(j);
      coef(j) = v.dot(r);
      r -= coef(j) * v;
    }
    // Saving coeficients
    matT.col(i).head(i + 1) = coef;

    // Reorthogonalization, if needed.
    if (r.norm() < 0.7 * norm_init) {
      // Second Gram Schmidt orthogonalization
      for (int j = 0; j < i + 1; ++j) {
        VectorXd v = basis.col(j);
        coef(j) = v.dot(r);
        r -= coef(j) * v;
      }
      matT.col(i).head(i + 1) = coef;
    }

    double r_norm = r.norm();
    matT(i + 1, i) = r_norm;

    // check for breakdown
    if (r_norm > tol_) {
      matQ.col(i + 1) = r / r_norm;
      VectorXd e_i = VectorXd::Zero(i + 1);
      e_i(i) = 1.0;
      // VectorXd v_aux = matT.block(1, 0, i + 1, i + 1).ldlt().solve(e_i);
      VectorXd v_aux =
          matT.block(1, 0, i + 1, i + 1).colPivHouseholderQr().solve(e_i);
      f = matQ.leftCols(i + 1) * v_aux;
    } else {
      // Deflation reset
      matT(i + 1, i) = 1.0;
      matQ.col(i + 1) = VectorXd::Zero(ndim_);
      VectorXd e_i = VectorXd::Zero(i + 1);
      e_i(i) = 1.0;
      // VectorXd v_aux = matT.block(1, 0, i + 1, i + 1).ldlt().solve(e_i);
      VectorXd v_aux =
          matT.block(1, 0, i + 1, i + 1).colPivHouseholderQr().solve(e_i);
      f = matQ.leftCols(i + 1) * v_aux;

      // Deflation verification
      VectorXd f_proj;
      for (int k : deflation) {
        VectorXd p = matP.col(k);
        double coef_f = p.dot(f) / p.dot(p);
        f_proj = f - coef_f * p;
      }

      if (f_proj.norm() > tol_) {
        deflation.push_back(i);
      } else {
        fmt::print("SOAR lucky breakdown.\n");
        break;
      }
    }
    matP.col(i + 1) = f;
  }

  fmt::print("zero1: {:9.3f}\n",
             (matQ.transpose() * matQ - MatrixXd::Identity(n, n)).norm());

  VectorXd e_n = VectorXd::Zero(n - 1);
  e_n(n - 2) = 1.0;
  VectorXd r = matA_ * matQ.col(n - 2) + matB_ * matP.col(n - 2);
  for (int i = 0; i < n - 1; ++i) {
    double coef = matQ.col(i).dot(r);
    r -= coef * matQ.col(i);
  }
  double nm = (matA_ * matQ.leftCols(n - 1) + matB_ * matP.leftCols(n - 1) -
               matQ.leftCols(n - 1) * matT.topLeftCorner(n - 1, n - 1) -
               r * e_n.transpose())
                  .norm();
  fmt::print("zero2: {:9.3f}\n", nm);
  return matQ;
}