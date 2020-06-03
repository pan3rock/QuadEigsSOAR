#include "soar.hpp"

#include <Eigen/Dense>

using namespace Eigen;

Soar::Soar(const Ref<const MatrixXd> matA, const Ref<const MatrixXd> matB,
           const Ref<const VectorXd> u)
    : matA_(matA), matB_(matB), u_(u), ndim_(matA.rows()) {}

void Soar::compute(int n) { ; }