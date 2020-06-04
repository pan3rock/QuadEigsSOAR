#ifndef QUADEIGS_H_
#define QUADEIGS_H_

#include <Eigen/Dense>
#include <memory>

class QuadEigs {
public:
  QuadEigs(const Eigen::Ref<const Eigen::MatrixXd> &matM,
           const Eigen::Ref<const Eigen::MatrixXd> &matD,
           const Eigen::Ref<const Eigen::MatrixXd> &matK);

  Eigen::VectorXcd eigenvalues(int m);

private:
  const int ndim_;
  Eigen::MatrixXd matM_, matD_, matK_;
  Eigen::MatrixXd matA_, matB_;
};

#endif