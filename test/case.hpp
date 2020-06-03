#ifndef TEST_CASE_H_
#define TEST_CASE_H_

#include <Eigen/Dense>

class Case1 {
public:
  Case1(int ndim);

  Eigen::VectorXcd alphas;
  Eigen::VectorXd betas;

private:
  Eigen::MatrixXd matM_, matD_, matK_;
  Eigen::MatrixXd matC_, matG_;
  const int ndim_;
};

#endif