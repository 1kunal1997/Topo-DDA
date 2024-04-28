#include <iostream>

#include "ObjDDAModel.h"
#include "math.h"

double SmoothDensity(double input, double ita, double beta) {
    double result = 0.0;
    if (input <= ita && input >= 0.0) {
        return ita * (exp(-beta * (1 - input / ita)) - (1 - input / ita) * exp(-beta));
    }
    else if (input > ita && input <= 1.0) {
        return (1 - ita) * (1 - exp(-beta * (input - ita) / (1 - ita)) + (input - ita) / (1 - ita) * exp(-beta)) + ita;
    }
    else {
        cout << "ERROR: FilterOption::SmoothDensity(double input)--input out of range" << endl;
        throw 1;
    }
}

ObjPointEDDAModel::ObjPointEDDAModel(vector<double> parameters, AProductCore* Core_, double d_, int N_, VectorXcd* P_, VectorXi* geometry_, double E0_, double K_, Vector3d n_E0_, Vector3d n_K_) {
    VectorXd PointEParameters = VectorXd::Zero(( parameters ).size( ));
    //auto it=(parameters).begin();
    for ( int i = 0; i <= int(( parameters ).size( ) - 1); i++ ) {
        PointEParameters(i) = parameters[ i ];
    }
    //Have_Penalty = HavePenalty_;
    x = PointEParameters(0);
    y = PointEParameters(1);
    z = PointEParameters(2);
    Have_Devx = false;
    Core = Core_;
    d = d_;
    N = N_;
    P = P_;
    geometry = geometry_;
    Vector3d n_E0 = n_E0_;
    Vector3d n_K = n_K_;
    double E0 = E0_;
    double K = K_;
    E_sum = Vector3cd::Zero( );                                                                             //ÊÇ²»ÊÇE_sumÍüÁË¼ÓE_extÁË£¿ It is actually in Rest.
    E_ext = Vector3cd::Zero( );
    E_ext(0) = E0 * n_E0(0) * ( cos(K * ( n_K(0) * x + n_K(1) * y + n_K(2) * z )) + sin(K * ( n_K(0) * x + n_K(1) * y + n_K(2) * z )) * 1i );
    E_ext(1) = E0 * n_E0(1) * ( cos(K * ( n_K(0) * x + n_K(1) * y + n_K(2) * z )) + sin(K * ( n_K(0) * x + n_K(1) * y + n_K(2) * z )) * 1i );
    E_ext(2) = E0 * n_E0(2) * ( cos(K * ( n_K(0) * x + n_K(1) * y + n_K(2) * z )) + sin(K * ( n_K(0) * x + n_K(1) * y + n_K(2) * z )) * 1i );
    // cout << R(3*5444+2) << "this" << endl;
}

void ObjPointEDDAModel::SingleResponse(int idx, bool deduction, bool hasPenalty){
    //VectorXcd P = model->get_P();
    //VectorXi R = model->get_R();
    double rx=x-d*(*geometry)(3*idx);                  //R has no d in it, so needs to time d
    double ry=y-d*(*geometry)(3*idx+1);
    double rz=z-d*(*geometry)(3*idx+2);
    //cout << rx << "," << ry << "," << rz << idx << endl;
    Matrix3cd A=(*Core).A_dic_generator(rx,ry,rz);
    if (deduction == false){
        E_sum(0)-=(A(0,0)*(*P)(3*idx)+A(0,1)*(*P)(3*idx+1)+A(0,2)*(*P)(3*idx+2));
        E_sum(1)-=(A(1,0)*(*P)(3*idx)+A(1,1)*(*P)(3*idx+1)+A(1,2)*(*P)(3*idx+2));
        E_sum(2)-=(A(2,0)*(*P)(3*idx)+A(2,1)*(*P)(3*idx+1)+A(2,2)*(*P)(3*idx+2));
    }
    else{
        E_sum(0)+=(A(0,0)*(*P)(3*idx)+A(0,1)*(*P)(3*idx+1)+A(0,2)*(*P)(3*idx+2));
        E_sum(1)+=(A(1,0)*(*P)(3*idx)+A(1,1)*(*P)(3*idx+1)+A(1,2)*(*P)(3*idx+2));
        E_sum(2)+=(A(2,0)*(*P)(3*idx)+A(2,1)*(*P)(3*idx+1)+A(2,2)*(*P)(3*idx+2));    
    }
}

void ObjPointEDDAModel::SingleResponseWithoutPenalty(int idx, bool deduction) {
    
}

double ObjPointEDDAModel::GroupResponse(){
    /*if (Have_Penalty){
        return (E_sum).norm()-(*evomodel).L1Norm();
    }*/
    /*else{*/
    return (E_sum).norm();
    /*}*/
    
}

double ObjPointEDDAModel::GetVal(){
    Reset();
    for (int idx=0;idx<N;idx++){
        SingleResponse(idx, false);
        // cout << E_sum(0) << endl;
    }
    return GroupResponse();
}

double ObjPointEDDAModel::GetValWithPenalty(double coeff) {
    return 0.0;
}

void ObjPointEDDAModel::Reset(){
    E_sum(0) = E_ext(0);
    E_sum(1) = E_ext(1);
    E_sum(2) = E_ext(2);
}



ObjIntegratedEDDAModel::ObjIntegratedEDDAModel(vector<double> parameters, int N_, VectorXcd* P_, VectorXi* geometry_, VectorXcd* al_) {
    VectorXd PointEParameters = VectorXd::Zero(( parameters ).size( ));
    //auto it=(parameters).begin();
    for ( int i = 0; i <= int(( parameters ).size( ) - 1); i++ ) {
        PointEParameters(i) = parameters[ i ];
    }
    //Have_Penalty = HavePenalty_;
    powNum = int(round(PointEParameters(0)));
    xMin = int(round(PointEParameters(1)));
    xMax = int(round(PointEParameters(2)));
    yMin = int(round(PointEParameters(3)));
    yMax = int(round(PointEParameters(4)));
    zMin = int(round(PointEParameters(5)));
    zMax = int(round(PointEParameters(6)));
    ita = PointEParameters(7);
    beta = PointEParameters(8);
    Have_Devx = true;

    N = N_;
    P = P_;
    geometry = geometry_;
    al = al_;
    penalty = 0.0;
    E_int = 0.0;
    namedebugfile = ".\\Squarecenter_4x4x1_it200_sym_eps0.1_penalty10\\debugfile.txt";
}

void ObjIntegratedEDDAModel::SingleResponse(int idx, bool deduction, bool hasPenalty) {

   // ofstream debug;
   // debug.open(namedebugfile, std::ios_base::app);

    if ((xMin <= (*geometry)(3 * idx) <= xMax)&&(yMin <= (*geometry)(3 * idx + 1) <= yMax)&&(zMin <= (*geometry)(3 * idx + 2) <= zMax)) {
        //double factor = SmoothDensity((*diel_old)(3 * idx), ita, beta);
        double factor = 1.0;
        double coeff = 10000.0;

        if (deduction == false) {
            E_int += factor * pow(abs((*al)(3 * idx) * (*P)(3 * idx)), powNum); // - penalty(i)
            E_int += factor * pow(abs((*al)(3 * idx + 1) * (*P)(3 * idx + 1)), powNum);
            E_int += factor * pow(abs((*al)(3 * idx + 2) * (*P)(3 * idx + 2)), powNum);

        }
        else {
            E_int -= factor * pow(abs((*al)(3 * idx) * (*P)(3 * idx)), powNum);
            E_int -= factor * pow(abs((*al)(3 * idx + 1) * (*P)(3 * idx + 1)), powNum);
            E_int -= factor * pow(abs((*al)(3 * idx + 2) * (*P)(3 * idx + 2)), powNum);

        }
       // debug << "\n";
       // cout << "\n" << endl;
    }
   // debug.close();
 
}

void ObjIntegratedEDDAModel::SingleResponseWithoutPenalty(int idx, bool deduction) {

    //ofstream debug;
    //debug.open(namedebugfile, std::ios_base::app);

    if ((xMin <= (*geometry)(3 * idx) <= xMax) && (yMin <= (*geometry)(3 * idx + 1) <= yMax) && (zMin <= (*geometry)(3 * idx + 2) <= zMax)) {
        //double factor = SmoothDensity((*diel_old)(3 * idx), ita, beta);
        double factor = 1;

        if (deduction == false) {
            E_int += factor * pow(abs((*al)(3 * idx) * (*P)(3 * idx)), powNum); // - penalty(i)
            E_int += factor * pow(abs((*al)(3 * idx + 1) * (*P)(3 * idx + 1)), powNum);
            E_int += factor * pow(abs((*al)(3 * idx + 2) * (*P)(3 * idx + 2)), powNum);

        }
        else {
            E_int -= factor * pow(abs((*al)(3 * idx) * (*P)(3 * idx)), powNum);
            E_int -= factor * pow(abs((*al)(3 * idx + 1) * (*P)(3 * idx + 1)), powNum);
            E_int -= factor * pow(abs((*al)(3 * idx + 2) * (*P)(3 * idx + 2)), powNum);

        }
        //debug << "\n";
    }
    //debug.close();
}

double ObjIntegratedEDDAModel::GroupResponseWithPenalty(double coeff) {

    double obj = fmax(1, E_int - coeff*penalty);
    return log(obj);
}

double ObjIntegratedEDDAModel::GroupResponse() {
    /*if (Have_Penalty){
        return (E_sum).norm()-(*evomodel).L1Norm();
    }*/
    /*else{*/
    return log(E_int);
    /*}*/

}

double ObjIntegratedEDDAModel::GetValWithPenalty(double coeff) {
    return 0;
}

double ObjIntegratedEDDAModel::GetVal() {
    Reset(); // E_int = 0.0
    for (int idx = 0; idx < N; idx++) {
        SingleResponse(idx, false);
        // cout << E_sum(0) << endl;
    }
    return GroupResponse(); // log(E_int)
}

void ObjIntegratedEDDAModel::Reset() {
    E_int = 0.0;
    penalty = 0.0;
}






