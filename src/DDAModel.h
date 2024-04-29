#ifndef TOPO_DDAMODEL_H_
#define TOPO_DDAMODEL_H_

#include "AProductCore.h"
#include "ObjDDAModel.h"
#include "FilterOption.h"

#include <list>
#include <vector>


using namespace std;
using namespace Eigen;

struct WeightPara {
    double weight;
    int position;
}; 

class DDAModel {
private:
    //------------------------------------Get from AProductCore------------------------------ For the same AMatrix, these are all the same
    AProductCore* Core;
    ObjDDAModel* objDDAModel;
    //AProductCore OwnCore;

    //-----------------------------------Not from AProductCore------------------------------- For the same AMatrix, these can be diff for diff DDAModel
    bool RResultSwitch;               //0(false) for plot only E field on the structure points (fast), 1(true) for using RResult different from R to plot (slow but adjustable).
    VectorXi RResult;                //The position matrix for the EResult (where you want to plot the E field)
    double E0;
    Vector3d n_E0;
    Vector3d n_K;
    VectorXcd P;
    VectorXcd E;
    VectorXcd Einternal;              //E field on structure points
    VectorXcd EResult;                //E field on designated points
    VectorXcd al;                       // 1 over alpha instead of alpha.
    VectorXcd diel;                     //Real dielectric from diel_old. Needed to calculate the Lorentz factor.
    bool verbose;
    VectorXcd P_max;
    VectorXcd al_max;

    int N;
    int Nx;
    int Ny;
    int Nz;
    double d;
    // geometry_values is an internal copy of the geometry, while geometry is actually a 
    // pointer. This causes segfaults if it is called in python bindings or in some other
    // setting where the data can go out of scope after initialization finishes.
    VectorXi geometry_values;
    VectorXi* geometry;
    //VectorXd* diel_old;
    double K;
    double lam;
    VectorXcd* material;
    //VectorXd* Para;
    int NFpara;

    vector<double> objPara;
    string objName;

    //------------------different for different angles------------------
    int time;
    int ITERATION;
    double Error;

    VectorXd dielectric_old;                //The 0~1 version of diel, 3*N
    VectorXd diel_old_max;

    // from StructureSpaceParahello
    VectorXi geometryPara;            //N dimension. N=number of dipoles. Each position stores the para index in VectorXi Para : 0->Para[0]...
    vector<vector<int>> Paratogeometry;
    VectorXd parameters;                    // THESE ARE THE INPUTDIELS !!! P dimension. P=number of parameters. Same as Para_origin if Filter=False. Filtered and biased para if Filter=True.
    VectorXd Para_origin;             //Un-filtered, unbiased para. No use when Filter=False.
    VectorXd Para_filtered;           //Filtered but unbiased para. No use when Filter=False.
    bool Filter;                      //True for with Filter. False for without Filter. Defualt as False for most initilizations.
    FilterOption* Filterstats;        //Only used when Filter=True
    vector<vector<WeightPara>> FreeWeight;
    bool Periodic;
    int Lm;
    int Ln;


public:
    DDAModel(string objName_, vector<double> objPara_, VectorXd* Para_, VectorXi* geometry_, VectorXd* diel_old_, int Nx_, int Ny_, int Nz_, int N_, Vector3d n_K_, double E0_, Vector3d n_E0_, double lam_, VectorXcd material_, double nback_, int MAXm_, int MAXn_, double Lm_, double Ln_, string AMatrixMethod_, double d_, bool verbose_ = true);
    DDAModel(double betamin_, double betamax_, double ita_, string betatype_, vector<int> filterIterations_, vector<double> filterRadii_, bool Filter_, string symmetry, vector<double> symaxis, bool Periodic_, string objName_, vector<double> objPara_, VectorXi* geometry_, VectorXd* Inputdiel, int Nx_, int Ny_, int Nz_, int N_, Vector3d n_K_, double E0_, Vector3d n_E0_, double lam_, VectorXcd material_, double nback_, int MAXm_, int MAXn_, double Lm_, double Ln_, string AMatrixMethod_, double d_, VectorXd sineIntegralValues_, VectorXd cosineIntegralValues_, double integralDelta_, bool verbose_ = true);
    ~DDAModel( );
    void bicgstab(int MAX_ITERATION, double MAX_ERROR);
    void bicgstab(int MAX_ITERATION, double MAX_ERROR, int EVOITERATION);  //FOR DEBUG ONLY. OUTPUT SOME VALUE AT CERTAIN EVO ITERATION.
    void change_E(VectorXcd E_);
    void reset_E();             //reset E to E0                                
    void UpdateAlpha();                                //update alpha according to updated diel in AProductCore.
    void UpdateAlphaSingle(int idx);
    void solve_E();                                                        //update the result E field on each dipole or on a designated space
    void update_E_in_structure();                                          //update the result E field on each dipole 
    VectorXcd Aproductwithalb(VectorXcd& b);                    //add the al*b term on base of AproductCore
    void output_to_file(string save_position, int iteration, int ModelLabel);              //especially used for EvoOptimization
    void output_to_file(string save_position, int iteration);             //For simplify output
    //void output_to_file(string save_position, double wavelength, int iteration);
    void InitializeP(VectorXcd& Initializer);
    VectorXcd* get_P();
    Vector3d get_nE0();
    Vector3d get_nK();
    double get_E0();
    VectorXcd* get_Einternal();
    AProductCore* get_Core();
    VectorXcd* get_al();
    VectorXcd* get_P_max();
    VectorXcd* get_al_max();
    int get_ITERATION();

    ObjDDAModel* ObjFactory(string ObjectName, vector<double> ObjectParameters);
    double calculateObjective( );
    bool get_HaveDevx( );
    void SingleResponse(int idx, bool deduction);
    double GroupResponse( );

    void solveElectricField(VectorXcd originPolarization, int BGS_MAX_ITER, double BGS_MAX_ERROR);

    void assignFreeWeightsForFilter( );
    void UpdateStr(VectorXd step, int current_it, int Max_it);
    void UpdateStrSingle(int idx, double value);
    void outputCStr_to_file(string save_position, int iteration, string mode = "normal");
    VectorXi* get_geometryPara( );
    VectorXd* get_parameters( );
    VectorXd get_parameter_copy( );
    VectorXd* get_Para_origin( );
    VectorXd* get_Para_filtered( );
    bool get_Filter( );
    FilterOption* get_Filterstats( );
    vector<vector<WeightPara>>* get_FreeWeight( );
    vector<vector<int>>* get_Paratogeometry( );
    VectorXd* get_dielectric_old( );
    VectorXd get_dielectrics_copy();
    VectorXd* get_diel_old_max( );

    tuple<VectorXd, VectorXcd> devx_and_Adevxp(double epsilon, double origin);
    VectorXcd devp(double epsilon, double origin);
    VectorXd calculateGradients(double epsilon_partial, double originalObjValue, int MAX_ITERATION, double MAX_ERROR);
    VectorXd gradients_filtered(VectorXd gradients, int current_it, int Max_it);
    void UpdateParameters(VectorXd step);

    //-----------------From AProductCore-----------------------

    int get_N();
    int get_Nx( );
    int get_Ny( );
    int get_Nz( );
    double get_lam( );
    double get_d();
    VectorXi* get_geometry();


    double get_beta( );
    double get_ita( );
    void update_beta(const int iteration, const int Max_iteration);
};

#endif