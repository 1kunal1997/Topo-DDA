#ifndef TOPO_SICI_H_
#define TOPO_SICI_H_

#include <Eigen/Core>

using namespace Eigen;

class SiCi {
    double disSi;
    double disCi;
    VectorXd Si;
    VectorXd Ci;
public:
    SiCi(VectorXd Si, VectorXd Ci, double delta);
    double get_Si(double x);
    double get_Ci(double y);
};

#endif
