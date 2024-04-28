#ifndef TOPO_APRODUCTCORE_H_
#define TOPO_APRODUCTCORE_H_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cufft.h"

#include <list>
#include <vector>
#include <Eigen/Core>

#include "Kernel.h"
#include "SiCi.h"

using namespace std;
using namespace Eigen;

class AProductCore {
private:
    //---------------------------------Necessary values for A matrix-----------------------------------------------------

    double K;
    double lam;
    double d;
    int N;

    //FFT related variables;
    double* AHos;                              //A_dicDoubl
    double* ADev;                              // double, but actually double*2 course real and imag are both stored in this double
    cufftDoubleComplex* A00, * A01, * A02, * A11, * A12, * A22; //The components in A for FFT, only in device
    double* bHos;                              //b in Aproduct
    double* bDev;                              // double, but actually double*2 course real and imag are both stored in this double
    cufftDoubleComplex* bxDev, * byDev, * bzDev; //b components for FFT, only in device
    int NxFFT, NyFFT, NzFFT;                   //2*(Nx, Ny, Nz) - 1
    int NFFT;                                  //NxFFT*NyFFT*NzFFT
    cufftDoubleComplex* Convx, * Convy, * Convz; //convolution of A and b on device

    //FFT plan for all
    cufftHandle Plan;

    //---------------------------------------Necessary for periodic A matrix----------------------------------------
    int MAXm;                            //-MAXm<=m<=MAXm
    int MAXn;                             //-MAXn<=n<=MAXn
    double Lm;                         //desplacement vector for one period, Currently should only be in x and y direction; d included do not need to time d
    double Ln;

    //--------------------------------Not necessary for A matrix but should be the same for diff DDAModel using the same A matrix------------------------------

    //VectorXcd diel;                   //real diel after 0~1 corresponds to real numbers
    VectorXcd material;
    //VectorXcd diel_max;                         //corresponds to the previous maximum obj
    double nback;                       //background material refractive index. 1.0 for air.

    //------------------------------For FCD and LDR choice-----------------------
    string AMatrixMethod;
    SiCi* SiCiValue;

    // For logging.
    bool verbose;

public:
    AProductCore(int Nx_, int Ny_, int Nz_, int N_, double d_, double lam_, VectorXcd material_, double nback_, int MAXm_, int MAXn_, double Lm_, double Ln_, string AMatrixMethod_, VectorXd sineIntegralValues, VectorXd cosineIntegralValues, double integralDelta, bool verbose);
    ~AProductCore();
    Matrix3cd A_dic_generator(double x, double y, double z);
    Matrix3cd A_dic_generator(double x, double y, double z, int m, int n);
    VectorXcd Aproduct(VectorXcd& b, VectorXi* R);                          //without al*b because al is in DDAModel and can be diff for the same AMatrix
    //void UpdateStr(VectorXd step);                   //alpha not updated because in DDAModel, do not forget!

    double get_lam();
    double get_K();
    VectorXcd* get_material();

    Matrix3cd FCD_inter(double x, double y, double z);
    Matrix3cd LDR_inter(double x, double y, double z);
};

#endif