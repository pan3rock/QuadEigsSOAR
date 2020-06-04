#ifndef SOAR_H_
#define SOAR_H_

#include <Eigen/Dense>
#include <vector>

class Soar {
public:
  Soar(const Eigen::Ref<const Eigen::MatrixXd> &matA,
       const Eigen::Ref<const Eigen::MatrixXd> &matB);

  Eigen::MatrixXd compute(int n);

private:
  const int ndim_;
  Eigen::MatrixXd matA_, matB_;
  Eigen::VectorXd u_;

  const double tol_ = 1.0e-10;
};

#endif