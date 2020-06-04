#ifndef TEST_CASE_H_
#define TEST_CASE_H_

#include <Eigen/Dense>

class Case1 {
public:
  Case1(const Eigen::Ref<const Eigen::MatrixXd> &matM,
        const Eigen::Ref<const Eigen::MatrixXd> &matD,
        const Eigen::Ref<const Eigen::MatrixXd> &matK);

private:
  const int ndim_;
  Eigen::MatrixXd matM_, matD_, matK_;
  Eigen::MatrixXd matC_, matG_;

public:
  Eigen::VectorXcd alphas;
  Eigen::VectorXd betas;
};

#endif