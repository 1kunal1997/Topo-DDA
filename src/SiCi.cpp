#include "SiCi.h"

using namespace std;

SiCi::SiCi(VectorXd Si_, VectorXd Ci_, double delta_) {
	disSi = delta_;
	disCi = delta_;
	Si = Si_;
	Ci = Ci_;
}

double SiCi::get_Si(double x) {
	int pos1 = floor(x / disSi) - 1;
	int pos2 = ceil(x / disSi) - 1;
	if (pos2 >= Si.size()){
		// We are about to index beyond the boundaries. The sine integral
		// converges very quickly to a constant value, so for large values of
		// x and c > 0, get_Si(x) is approximately equal to get_Si(x + c). In
		// this case, we return the last entry in Si.
		return Si(Eigen::placeholders::last);
	}
	// otherwise, interpolate.
	double val1 = Si(pos1);
	double val2 = Si(pos2);
	double axis1 = (pos1 + 1) * disSi;
	double axis2 = (pos2 + 1) * disSi;
	return val1 + ((val2 - val1) / (axis2 - axis1)) * (x - axis1);
}

double SiCi::get_Ci(double x) {
	int pos1 = floor(x / disCi) - 1;
	int pos2 = ceil(x / disCi) - 1;
	if (pos2 >= Ci.size()){
		// We are about to index beyond the boundaries. The cosine integral also
		// converges very quickly, so we can return the last entry in Ci.
		return Ci(Eigen::placeholders::last);
	}
	// otherwise, interpolate.
	double val1 = Ci(pos1);
	double val2 = Ci(pos2);
	double axis1 = (pos1 + 1) * disCi;
	double axis2 = (pos2 + 1) * disCi;
	return val1 + ((val2 - val1) / (axis2 - axis1)) * (x - axis1);
}
