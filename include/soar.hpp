#ifndef SOAR_H_
#define SOAR_H_

#include <Eigen/Dense>

class Soar {
public:
  Soar(const Eigen::Ref<const Eigen::MatrixXd> matA,
       const Eigen::Ref<const Eigen::MatrixXd> matB,
       const Eigen::Ref<const Eigen::VectorXd> u);

  void compute(int n);

private:
  Eigen::MatrixXd matA_, matB_;
  Eigen::VectorXd u_;

  const int ndim_;
  const double tol_ = 1.0e-10;
};

#endif