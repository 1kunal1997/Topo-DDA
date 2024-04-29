#define NUM_THREADS 6

#include <chrono>
#include <iostream>
#include <fstream>
#include <map>

#include "DDAModel.h"
#include "Tools.h"

using namespace std::chrono;

 // namespace

ObjDDAModel* DDAModel::ObjFactory(string ObjectName, vector<double> ObjectParameters) {
    /*if (HavePenalty) {
        cout << "Using L1 Penalty with Penalty Factor " << PenaltyFactor << endl;
    }*/
    /*if ( objName == "PointE" ) {
        return new ObjPointEDDAModel(ObjectParameters, ObjDDAModel);
    }*/

    // TODO: Fix the fact that this switches on an internally defined (in the header)
    // class member variable objName, NOT ObjectName, and therefore always falls through to
    // the default case.

    if ( objName == "IntegratedE" ) {
        return new ObjIntegratedEDDAModel(ObjectParameters, N, &P, geometry, &al);
    }

    if (verbose){
        cout << "["<< objName << "] is not a supported objective name "
             << "defaulting to [IntegratedE]" << endl;
    }
    return new ObjIntegratedEDDAModel(ObjectParameters, N, &P, geometry, &al);
}
void FCurrentinsert(map<vector<int>, int>* FCurrent, vector<int> currentxy, int* currentpos, string insertmode, vector<double>* symaxis) {
    if ( insertmode == "None" ) {
        ( *FCurrent ).insert(pair<vector<int>, int>(currentxy, *currentpos));
        ( *currentpos ) += 1;
    }
    else if ( insertmode == "4fold" ) {
        vector<int> sym1{ int(round(2 * ( *symaxis )[ 0 ] - currentxy[ 0 ])), currentxy[ 1 ] };
        vector<int> sym2{ int(round(2 * ( *symaxis )[ 0 ] - currentxy[ 0 ])), int(round(2 * ( *symaxis )[ 1 ] - currentxy[ 1 ])) };
        vector<int> sym3{ currentxy[ 0 ], int(round(2 * ( *symaxis )[ 1 ] - currentxy[ 1 ])) };
        if ( ( *FCurrent ).count(sym1) ) {
            ( *FCurrent ).insert(pair<vector<int>, int>(currentxy, ( *FCurrent )[ sym1 ]));
        }
        else if ( ( *FCurrent ).count(sym2) ) {
            ( *FCurrent ).insert(pair<vector<int>, int>(currentxy, ( *FCurrent )[ sym2 ]));
        }
        else if ( ( *FCurrent ).count(sym3) ) {
            ( *FCurrent ).insert(pair<vector<int>, int>(currentxy, ( *FCurrent )[ sym3 ]));
        }
        else {
            ( *FCurrent ).insert(pair<vector<int>, int>(currentxy, *currentpos));
            ( *currentpos ) += 1;
        }
    }
    else {
        cout << "FCurrentinsert: This sym mode not supported yet" << endl;
        throw 1;
    }
}

double calweight(int xo, int yo, int x, int y, double r) {
    return r - sqrt(double(x - xo) * double(x - xo) + double(y - yo) * double(y - yo));
}

bool circlerange(int xo, int yo, int x, int y, double r) {
    if ( double(x - xo) * double(x - xo) + double(y - yo) * double(y - yo) <= r * r - 0.01 ) {
        return true;
    }
    else {
        return false;
    }
}

DDAModel::DDAModel(double betamin_, double betamax_, double ita_, string betatype_, vector<int> filterIterations_, vector<double> filterRadii_, bool Filter_, string symmetry, vector<double> symaxis, bool Periodic_, string objName_, vector<double> objPara_, VectorXi* geometry_, VectorXd* Inputdiel, int Nx_, int Ny_, int Nz_, int N_, Vector3d n_K_, double E0_, Vector3d n_E0_, double lam_, VectorXcd material_, double nback_, int MAXm_, int MAXn_, double Lm_, double Ln_, string AMatrixMethod_, double d_, VectorXd sineIntegralValues_, VectorXd cosineIntegralValues_, double integralDelta_, bool verbose_) {
    /*

    // Note: Geometry was previously passed in as a pointer, but in the PythonBindings, it gets called with a temporary copy of the input
    // this causes a dangling pointer, which leads to a segfault whenever the geometry is accessed next.
    // While the ideal solution would be to wrap an Eigen::Vector reference (so that we have only one copy of the matrix between python and cpp)
    // For now, we will just make DDAModel own its own geometry, so that the geometry pointer can point to the internal copy.

    Arguments:
        
        betamin_, betamax_, ita_, betatype: Used to update beta, which is used in the SmoothDensity function

        filterIterations_: vector of ints (size determined by user) for which iteration to change the filter radius. 
            used to construct 'filterinfo' struct.

        filterRadii_: one-to-one mapping to 'filterIterations_' for setting the radii.

        Filter_: boolean that determines whether you are using the internal Filter or not.

        symmetry: string to turn on or off "4-fold" symmetry. to add other options for symmetry, look up 'dividesym'.

        symaxis: vector of length 2 that sets the x and y coordinates for the axes of symmetry.

        Periodic_: boolean for whether you want a periodic structure (in xy) or not.

        objName_: name of the objective function to use.

        objPara_: set of parameters needed for the corresponding obj function. size depends on obj function.

        geometry_: vector of ints of length 3*N. numbering system for each pixel. (0, 0, 0, 1, 0, 0, 2, 0, 0, ...)

        InputDiel_: vector of doubles of length 3*N. initialization of the pixel values (parameters)

        Nx_, Ny_, Nz_,: ints for the number of pixels in the x, y, and z direction. 

        N_: total number of pixels in structure

        n_K_: vector of doubles of length 3. direction that incoming plane wave of light is propagating in.

        n_E0_: vector of doubles of length 3. Defines the polarization of light.

        E0_: amplitude of the incoming plane wave.

        lam_: wavelength of the incoming plane wave.

        material_: vector of complex doubles of length 2. first element is the dielectric function of external domain,
            second element is the dielectric function of the material of interest. determined using interpolation.

        nback_: refractive index of background.

        MAXm_, MAXn_: integers used in AProductCore for periodicity reasons.

        Lm_, Ln_: integers for periodic boundary in the x and y direction, respectively.

        AMatrixMethod_: string used to determine which method to use in AProductCore.

        d_: size of pixel in nanometers.

    */
    verbose = verbose_;
    if (verbose){
        cout<<"Object name: " << objName_ << endl;
        cout<<"betatype: " << betatype_ << endl;
    }

    d = d_;
    Filter = Filter_;
    // TODO: geometry is a pointer. geometry_values is an internal copy that stores the values.
    geometry_values = VectorXi(geometry_->size());
    for (int i = 0; i < geometry_->size(); i++){
        geometry_values(i) = (*geometry_)(i); // disgusting.
    }
    geometry = &geometry_values;
    if (verbose){
        cout<<"Geometry size: "<<geometry->size( )<<endl;
    }
    Nx = Nx_;
    Ny = Ny_;
    Nz = Nz_;
    N = N_;
    Periodic = Periodic_;
    Lm = Lm_;
    Ln = Ln_;

    int dividesym;
    if ( symmetry == "None" ) {
        dividesym = 1;
    }
    else if ( symmetry == "4fold" ) {
        dividesym = 4;
    }
    else {
        cout << "SpacePara: not None nor 4 fold. Not supported" << endl;
        throw 1;
    }

    vector<filterinfo> rfilter;

    for ( int i = 0; i < filterIterations_.size( ); i++ ) {
        rfilter.push_back(filterinfo{ filterIterations_[i], filterRadii_[i]});
    }

    NFpara = int(round(( int(geometry->size( )) / 3 / Nz / dividesym )));    // number of free parameters. for extruded, symmetric, take one quadrant of one xy-plane of geo. 121 in standard case
    if (verbose) {
        cout << "NFpara " << NFpara << endl;
    }

    parameters = VectorXd::Zero(NFpara);
    geometryPara = VectorXi::Zero(N);

    // stores an index that can be used to find the free parameter index associated with that index.
    // uses symmetry and reflections in FCurrentInsert to keep the range [0,120]. for example, 
    // geometryPara(Nx-1) = 1, geometry(Nx) = 0, geometry(Nx*Ny+1)=0 because of symmetry/extrusions.
    geometryPara = VectorXi::Zero(N);
    map<vector<int>, int> FCurrent;         // only used to design geometryPara, so can be removed if geometryPara designed differently
    int currentpos = 0;
    for ( int i = 0; i <= N - 1; i++ ) {
        int x = ( *geometry ) ( 3 * i );
        int y = ( *geometry ) ( 3 * i + 1 );

        vector<int> currentxy{ x,y };

        // for extruded geometries, same xy position can be used to refer to pixels that have different z-coord.
        if ( !FCurrent.count(currentxy) ) {

            FCurrentinsert(&FCurrent, currentxy, &currentpos, symmetry, &symaxis);
        }
        geometryPara(i) = FCurrent[ currentxy ];
        //cout << "geometryPara(" << i << ") is: " << geometryPara(i) << endl;
    }

    // This is used to map a free parameter position (0 to NFPara, or 121) to a vector of
    // positions that this free parameter maps to, considering relfections and extrusions.
    // for example, the first vector would be (0, Nx, Nx*Ny-Nx, Nx*Ny, ...). if Nz is 10,
    // and you have symmetry, each position will map to 10*4, or 40 other pixels.
    Paratogeometry = vector<vector<int>>(NFpara);
    for ( int i = 0; i <= N - 1; i++ ) {
        //cout << "dipole position : " << i << endl;

        ( Paratogeometry[ geometryPara(i) ] ).push_back(i);

        /*vector<int> currentvector = Paratogeometry[geometryPara(i)];
        for (int j = 0; j < currentvector.size(); j++) {
            cout << "Paratogeometry at index " << j << " is: " << currentvector[j] << endl;
        } */

    }

    // used to map a pixel vector to its inputdiel value (0-1)
    map<vector<int>, double> Inputmap;
    if ( geometry->size( ) != ( *Inputdiel ).size( ) ) {
        cout << "ERROR: SpacePara::SpacePara: Filter==(*InputGeo).size() != (*Inputdiel).size()" << endl;
        throw 1;
    }
    int Inputsize = int(round(int( geometry->size( )) / 3));
    for ( int i = 0; i < Inputsize; i++ ) {
        Inputmap.insert(pair<vector<int>, double>(vector<int>{( *geometry ) ( 3 * i ), ( *geometry ) ( 3 * i + 1 ), ( *geometry ) ( 3 * i + 2 )}, ( *Inputdiel )( 3 * i )));
    }

    // used to fill Para, which are the parameter values of the free indices (NFpara)
    // or one quadrant in a symmetric, extruded structure. using Paratogeometry here
    // to fetch each position, but if this is the only use of Paratogeometry, seems useless.
    for ( int i = 0; i < NFpara; i++ ) {

        int pos = Paratogeometry[ i ][ 0 ];
        //cout << "Pos at position " << i << " is: " << pos << endl;
        vector<int> node{ ( *geometry ) ( 3 * pos ), ( *geometry ) ( 3 * pos + 1 ), ( *geometry ) ( 3 * pos + 2 ) };
        parameters(i) = Inputmap[ node ];
    }

    if (verbose) {
        cout << "para values: " << endl;
        for ( int i = 0; i < parameters.size( ); i++ ) {
            cout << parameters(i) << " ";
        } cout << endl;
    }

    if ( Filter == true ) {
        Para_origin = parameters;
        Para_filtered = parameters;
        Filterstats = new FilterOption(betamin_, betamax_, ita_, betatype_, rfilter);
        if ( Filterstats == NULL ) {
            cout << "ERROR: SpacePara::SpacePara: Filter==true then Filterstats must be passed in." << endl;
            throw 1;
        }

        assignFreeWeightsForFilter( );

    }

    // original CoreStructure stuff below

    //---------------------------------------------------initial diel------------------------------------
    dielectric_old = VectorXd::Zero(3 * N);
    diel_old_max = dielectric_old;
    for ( int i = 0; i <= N - 1; i++ ) {
        double dieltmp = parameters(geometryPara(i));
        dielectric_old(3 * i) = dieltmp;
        dielectric_old(3 * i + 1) = dieltmp;
        dielectric_old(3 * i + 2) = dieltmp;
    }

    Core = new AProductCore(Nx_, Ny_, Nz_, N_, d_, lam_, material_, nback_, MAXm_, MAXn_, Lm_ * d_, Ln_ * d_, AMatrixMethod_, sineIntegralValues_, cosineIntegralValues_, integralDelta_, verbose_);
    time = 0;
    ITERATION = 0;
    Error = 0.0;
    E0 = E0_;
    n_K = n_K_;
    n_E0 = n_E0_;

    N = N_;
    Nx = Nx_;
    Ny = Ny_;
    Nz = Nz_;
    lam = lam_;
    if (verbose){
        cout << "lam in DDAModel is: " << lam << endl;
    }
    double nback = nback_;
    K = Core->get_K( );
    d = d_;
    material = Core->get_material( );


    objDDAModel = ObjFactory(objName_, objPara_);
    RResultSwitch = false;
    RResult = *geometry;

    P = VectorXcd::Zero(N * 3);
    P_max = P;
    E = VectorXcd::Zero(N * 3);
    Einternal = VectorXcd::Zero(N * 3);
    EResult = VectorXcd::Zero(N * 3);
    for ( int i = 0; i < N; i++ ) {
        E(3 * i) = E0 * n_E0(0) * ( cos(K * d * ( n_K(0) * ( *geometry )( 3 * i ) + n_K(1) * ( *geometry )( 3 * i + 1 ) + n_K(2) * ( *geometry )( 3 * i + 2 ) )) + sin(K * d * ( n_K(0) * ( *geometry )( 3 * i ) + n_K(1) * ( *geometry )( 3 * i + 1 ) + n_K(2) * ( *geometry )( 3 * i + 2 ) )) * 1i );
        E(3 * i + 1) = E0 * n_E0(1) * ( cos(K * d * ( n_K(0) * ( *geometry )( 3 * i ) + n_K(1) * ( *geometry )( 3 * i + 1 ) + n_K(2) * ( *geometry )( 3 * i + 2 ) )) + sin(K * d * ( n_K(0) * ( *geometry )( 3 * i ) + n_K(1) * ( *geometry )( 3 * i + 1 ) + n_K(2) * ( *geometry )( 3 * i + 2 ) )) * 1i );
        E(3 * i + 2) = E0 * n_E0(2) * ( cos(K * d * ( n_K(0) * ( *geometry )( 3 * i ) + n_K(1) * ( *geometry )( 3 * i + 1 ) + n_K(2) * ( *geometry )( 3 * i + 2 ) )) + sin(K * d * ( n_K(0) * ( *geometry )( 3 * i ) + n_K(1) * ( *geometry )( 3 * i + 1 ) + n_K(2) * ( *geometry )( 3 * i + 2 ) )) * 1i );
    }
    al = VectorXcd::Zero(N * 3);
    diel = VectorXcd::Zero(N * 3);
    for ( int i = 0; i < N * 3; i++ ) {
        int labelfloor = int(floor(dielectric_old( i )));
        int labelnext = labelfloor + 1;
        if ( labelfloor >= 1 ) {
            labelnext = labelfloor;
        }
        std::complex<double> diel_tmp = ( *material )( labelfloor ) + (dielectric_old( i ) - double(labelfloor) ) * ( ( *material )( labelnext ) - ( *material )( labelfloor ) );
        diel(i) = diel_tmp;
        al(i) = 1.0 / Get_Alpha(lam, K, d, diel_tmp, n_E0, n_K);
    }

    //cout << "al" << al(0) << endl;

    al_max = al;
}

DDAModel::~DDAModel( ) {
    delete Core;
    Core = nullptr;
}

double PtoFderivative(const double input, const double beta, const double ita) {
    double result = 0.0;
    if ( input <= ita && input >= 0.0 ) {
        return beta * exp(-beta * ( 1 - input / ita )) + exp(-beta);
    }
    else if ( input > ita && input <= 1.0 ) {
        return beta * exp(-beta * ( input - ita ) / ( 1 - ita )) + exp(-beta);
    }
    else {
        cout << "ERROR: PtoFderivative(const double input, const double beta, const double ita)--input out of range" << endl;
        throw 1;
    }
}

VectorXd DDAModel::gradients_filtered(VectorXd gradients, int current_it, int Max_it) {

    Filterstats->update_beta(current_it, Max_it);                     //current_it is actually the it in current evo-1 as the str is updated in iteration-1.
    const double gbeta = Filterstats->get_beta( );
    const double gita = Filterstats->get_ita( );

    VectorXd result = VectorXd::Zero(NFpara);
    if ( NFpara != FreeWeight.size( ) ) {
        cout << "ERROR: EvoDDAModel::gradients_filtered--NFpara != (*FreeWeight).size()" << endl;
        throw 1;
    }
    for ( int i = 0; i <= NFpara - 1; i++ ) {
        //int position = (*Free)(i);
        //double pftmp = (*Para_filtered)(position);
        //double ptofd = PtoFderivative(pftmp, gbeta, gita);

        int num_weight = ( FreeWeight[ i ] ).size( );
        double sum_weight = 0.0;
        for ( int j = 0; j <= num_weight - 1; j++ ) {
            double weight = FreeWeight[ i ][ j ].weight;
            int weightpos = FreeWeight[ i ][ j ].position;
            sum_weight += weight;
        }
        for ( int j = 0; j <= num_weight - 1; j++ ) {
            double weight = FreeWeight[ i ][ j ].weight;
            int weightpos = FreeWeight[ i ][ j ].position;
            if ( weightpos < NFpara ) {                        //If there are non-paras also weighted but do not need to calculate gradients to.
                result(weightpos) += ( weight / sum_weight );
            }

        }
    }

    for ( int i = 0; i <= NFpara - 1; i++ ) {
        int position = i;
        double pftmp = Para_filtered( position );
        double ptofd = PtoFderivative(pftmp, gbeta, gita);
        result(position) = result(position) * ptofd * gradients(position);
    }

    return result;

}

VectorXd DDAModel::calculateGradients(double epsilon_partial, double originalObjValue, int MAX_ITERATION, double MAX_ERROR) {

    VectorXd gradients = VectorXd::Zero(NFpara);

    if (verbose){
        cout << "Computing partial derivatives." << endl;
        cout << "---------------------------START PARTIAL DERIVATIVE ----------------------" << endl;
    }

    //----------------------------------------get partial derivative of current model---------------------------
    high_resolution_clock::time_point t1 = high_resolution_clock::now( );
    VectorXd devx;
    VectorXcd Adevxp;
    VectorXcd devp;
    tie(devx, Adevxp) = this->devx_and_Adevxp(epsilon_partial, originalObjValue);
    //tie(devx, Adevxp) = this->devx_and_Adevxp(epsilon_partial, Model, objfunc, originalObjValue);
    if (verbose){
        cout << "devx and adevxp finished, starting devp" << endl;
    }
    devp = this->devp(epsilon_partial, originalObjValue);
    if (verbose){
        cout << "done with devp, changing E using devp" << endl;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now( );
    auto duration = duration_cast< milliseconds >( t2 - t1 ).count( );
    if (verbose){
        cout << "------------------------PARTIAL DERIVATIVE finished in " << duration / 1000 << " s-------------------------" << endl;
        cout << "---------------------------START ADJOINT PROBLEM ----------------------" << endl;
    }
    //------------------------------------Solving adjoint problem-----------------------------------------

    change_E(devp);
    VectorXcd Ptmp = VectorXcd::Zero(N * 3);
    InitializeP(Ptmp);
    bicgstab(MAX_ITERATION, MAX_ERROR);

    VectorXcd lambdaT = P;
    reset_E( );                                  //reset E to initial value
    //Adjointiterations << ( *Model ).get_ITERATION( ) << endl;
    //TotalAdjointIt += ( *Model ).get_ITERATION( );
    if (verbose) {
        cout << "Done with adjoint problem." << endl;
    }
    //times lambdaT and Adevxp together
    VectorXcd mult_result;
    mult_result = VectorXcd::Zero(NFpara);               //multiplication result has the length of parameter

    for ( int i = 0; i <= NFpara - 1; i++ ) {
        int FreeParaPos = i;

        vector<int>::iterator it = Paratogeometry[ FreeParaPos ].begin( );
        for ( int j = 0; j <= Paratogeometry[ FreeParaPos ].size( ) - 1; j++ ) {
            int position = *it;
            mult_result(i) += lambdaT(3 * position) * Adevxp(3 * position);
            mult_result(i) += lambdaT(3 * position + 1) * Adevxp(3 * position + 1);
            mult_result(i) += lambdaT(3 * position + 2) * Adevxp(3 * position + 2);
            it++;
        }
    }
    if (verbose) {
        cout << "Done multiplying lambdaT and Adevxp." << endl;
    }
    VectorXd mult_result_real = VectorXd::Zero(NFpara);
    for ( int i = 0; i <= NFpara - 1; i++ ) {
        complex<double> tmp = mult_result(i);
        mult_result_real(i) = tmp.real( );
    }
    gradients = devx - mult_result_real;              //What's the legitimacy in here to ignore the imag part?

    return gradients;

}

tuple<VectorXd, VectorXcd> DDAModel::devx_and_Adevxp(double epsilon, double origin) {

    VectorXcd Adevxp = VectorXcd::Zero(3 * N);
    VectorXd devx = VectorXd::Zero(NFpara);

    if (verbose){
        cout << "NFpara is: " << NFpara << endl;
    }

    for ( int i = 0; i < NFpara; i++ ) {
        int FreeParaPos = i;
        double diel_old_origin = parameters(FreeParaPos);
        double diel_old_tmp = diel_old_origin;
        int sign = 0;
        if ( diel_old_origin >= epsilon ) {
            sign = -1;
        }
        else {
            sign = 1;
        }
        diel_old_tmp += sign * epsilon;

        vector<int>::iterator it = Paratogeometry[ FreeParaPos ].begin( );
        for ( int j = 0; j <= Paratogeometry[ FreeParaPos ].size( ) - 1; j++ ) {
            //cout << (*it) << endl;
            int position = *it;
            complex<double> alphaorigin = al( 3 * position );

            if ( get_HaveDevx( ) )
                SingleResponse(position, true);
            UpdateStrSingle(position, diel_old_tmp);
            UpdateAlphaSingle(position);

            if ( get_HaveDevx( ) )
                SingleResponse(position, false);
            complex<double> change = ( al( 3 * position ) - alphaorigin ) / ( sign * epsilon );
            Adevxp(3 * position) = change;
            Adevxp(3 * position + 1) = change;
            Adevxp(3 * position + 2) = change;

            it++;
        }

        devx(i) = ( GroupResponse( ) - origin ) / ( sign * epsilon );  //If some obj has x dependency but you denote the havepenalty as false, it will still actually be calculated in an efficient way.
        it = Paratogeometry[ FreeParaPos ].begin( );
        for ( int j = 0; j <= Paratogeometry[ FreeParaPos ].size( ) - 1; j++ ) {
            int position = *it;
            if ( get_HaveDevx( ) )
                SingleResponse(position, true);

            UpdateStrSingle(position, diel_old_origin);
            UpdateAlphaSingle(position);

            if ( get_HaveDevx( ) )
                SingleResponse(position, false);
            it++;
        }

    }

    for ( int i = 0; i <= 3 * N - 1; i++ ) {
        Adevxp(i) = Adevxp(i) * P(i);
    }

    return make_tuple(devx, Adevxp);
}

VectorXcd DDAModel::devp(double epsilon, double origin) {
    //move origin=Obj0->GetVal() outside because it is the same for one partial derivative of the entire structure

    VectorXcd result = VectorXcd::Zero(P.size()); // 3N dimension

    for ( int i = 0; i <= P.size( ) - 1; i++ ) {
        int position = i / 3;

        SingleResponse(position, true);

        P(i) = P(i) + epsilon;

        SingleResponse(position, false);

        result(i) += ( GroupResponse() - origin ) / epsilon;

        SingleResponse(position, true);

        P(i) = P(i) - epsilon;

        complex<double> epsilonimag = epsilon * 1.0i;

        P(i) = P(i) + epsilonimag;

        SingleResponse(position, false);

        complex<double> tmpRes = ( GroupResponse() - origin ) / epsilonimag;
        result(i) += tmpRes;

        SingleResponse(position, true);

        P(i) = P(i) - epsilonimag;

        SingleResponse(position, false);
    }

    if (verbose) {
        cout << "Devp_sum: " << result.sum( ) << endl;
    }
    return result;
}

void DDAModel::assignFreeWeightsForFilter( ) {
    //Only works when freepara is 2D binding (Only do the filter in 2D)
    int NFpara = parameters.size( );
    FreeWeight = vector<vector<WeightPara>>(NFpara);

    double rfilter = ( *Filterstats ).get_rfilter( );
    for ( int i = 0; i <= NFpara - 1; i++ ) {
        int poso = Paratogeometry[ i ][ 0 ];                 //As 2D extrusion is assumed, different z does not matter
        int xo = ( *geometry ) ( 3 * poso );
        int yo = ( *geometry ) ( 3 * poso + 1 );
        int zo = ( *geometry ) ( 3 * poso + 2 );



        for ( int j = 0; j <= NFpara - 1; j++ ) {
            //bool inornot = false;
            //cout << j << endl;
            for ( int k = 0; k < Paratogeometry[ j ].size( ); k++ ) {
                int posr = Paratogeometry[ j ][ k ];

                int xr = ( *geometry ) ( 3 * posr );
                int yr = ( *geometry ) ( 3 * posr + 1 );
                int zr = ( *geometry ) ( 3 * posr + 2 );
                if ( Periodic == false ) {
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr, yr, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr, yr, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                }
                else {
                    //Own cell
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr, yr, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr, yr, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                    //0, -1
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr, yr - Ln, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr, yr - Ln, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                    //-1, -1
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr - Lm, yr - Ln, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr - Lm, yr - Ln, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                    //-1, 0
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr - Lm, yr, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr - Lm, yr, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                    //-1, 1
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr - Lm, yr + Ln, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr - Lm, yr + Ln, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                    //0, 1
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr, yr + Ln, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr, yr + Ln, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                    //1, 1
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr + Lm, yr + Ln, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr + Lm, yr + Ln, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                    //1, 0
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr + Lm, yr, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr + Lm, yr, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                    //1, -1
                    if ( ( zo == zr ) && ( circlerange(xo, yo, xr + Lm, yr - Ln, rfilter) ) ) {
                        //1. Same xy plane 2. inside the circle in xy plane 
                        //para>=2 wont be in NFpara
                        int posweight = j;
                        double weight = calweight(xo, yo, xr + Lm, yr - Ln, rfilter);
                        FreeWeight[ i ].push_back(WeightPara{ weight,posweight });
                        //break;
                        //As soon as one in the entire z direction is verified, no need for looking at others for 1 j. But when there is symmetry, this is needed because same z can have differnt x, y.
                    }
                }


            }

        }
    }
}

void DDAModel::UpdateParameters(VectorXd step) {
    if (verbose){
        cout << "step in UpdateParameters: " << step.mean( ) << endl;
    }

    int Parasize = parameters.size( );
    if ( Parasize != step.size( ) ) {
        cout << "ERROR: In CoreStructure::UpdateStr(VectorXd step), step.size!=FreePara.size";
        throw 1;
    }

    for ( int i = 0; i <= Parasize - 1; i++ ) {
        parameters(i) += step(i);
        if ( parameters(i) >= 1 ) {
            parameters(i) = 1;
        }
        if ( parameters(i) <= 0 ) {
            parameters(i) = 0;
        }
    }

    for ( int i = 0; i <= N - 1; i++ ) {
        int position = geometryPara(i);
        double value = parameters(position);
        dielectric_old(3 * i) = value;
        dielectric_old(3 * i + 1) = value;
        dielectric_old(3 * i + 2) = value;
    }

    UpdateAlpha( );

}

void DDAModel::UpdateStr(VectorXd step, int current_it, int Max_it) {
    if (verbose){
        cout << "step in UpdateStr: " << step.mean( ) << endl;
    }

    int Parasize = parameters.size( );
    if ( Parasize != step.size( ) ) {
        cout << "ERROR: In CoreStructure::UpdateStr(VectorXd step), step.size!=FreePara.size";
        throw 1;
    }

    if ( Filter == true ) {
        //When there is filter
        ( *Filterstats ).update_beta(current_it, Max_it);                  //Update beta value according to current iteration

        for ( int i = 0; i <= Parasize - 1; i++ ) {
            Para_origin(i) += step(i);
            if ( Para_origin(i) >= 1 ) {
                Para_origin(i) = 1;
            }
            if ( Para_origin(i) <= 0 ) {
                Para_origin(i) = 0;
            }
        }
        if (verbose){
            cout << "Beta at iteration " << current_it << " is " << Filterstats->get_beta( ) << endl;
        }

        for ( int i = 0; i <= Parasize - 1; i++ ) {
            int weightnum = ( FreeWeight[ i ] ).size( );
            double numerator = 0.0;
            double denominator = 0.0;
            for ( int j = 0; j <= weightnum - 1; j++ ) {
                numerator += ( FreeWeight[ i ][ j ].weight ) * Para_origin(FreeWeight[ i ][ j ].position);
                denominator += ( FreeWeight[ i ][ j ].weight );

            }
            Para_filtered(i) = numerator / denominator;

            double Para_physical = Filterstats->SmoothDensity(Para_filtered(i));
            parameters(i) = Para_physical;

        }
    }
    else {//When there is no filter
        for ( int i = 0; i <= Parasize - 1; i++ ) {
            parameters(i) += step(i);
            if ( parameters(i) >= 1 ) {
                parameters(i) = 1;
            }
            if ( parameters(i) <= 0 ) {
                parameters(i) = 0;
            }
        }
    }


    for ( int i = 0; i <= N - 1; i++ ) {
        int position = geometryPara(i);
        double value = parameters(position);
        dielectric_old(3 * i) = value;
        dielectric_old(3 * i + 1) = value;
        dielectric_old(3 * i + 2) = value;
    }
}


void DDAModel::UpdateStrSingle(int idx, double value) {

    dielectric_old(3 * idx) = value;
    dielectric_old(3 * idx + 1) = value;
    dielectric_old(3 * idx + 2) = value;

}

void DDAModel::outputCStr_to_file(string save_position, int iteration, string mode) {

    if ( mode == "normal" ) {
        string name;
        name = save_position + "CoreStructure" + to_string(iteration) + ".txt";
        ofstream fout(name);
        fout << Nx << endl << Ny << endl << Nz << endl << N << endl;
        fout << &geometry << endl;
        fout << dielectric_old << endl;
        fout << d << endl;
        fout.close( );
    }
    else {
        string name;
        name = save_position + "CoreStructure" + to_string(iteration) + ".txt";
        ofstream fout(name);
        fout << dielectric_old << endl;
        fout.close( );
    }
}

double DDAModel::calculateObjective( ) {
    return objDDAModel->GetVal( );
}

void DDAModel::solveElectricField(VectorXcd originPolarization, int BGS_MAX_ITER, double BGS_MAX_ERROR){
    // This has to be called before each optimization iteration because otherwise, we will not
    // initialize many fields that are necessary to compute the objective and gradients in the new structure.
    // Note: originPolarization might be able to be preconditioned (e.g. using previous value of the P matrix).
    // However, this has always been set to zeros in our experiments.
    // originPolarization: This is a vector of size 3*N (where N is the total number of pixels). 
    // TODO: Overhaul code to make all the 3*N overheads go to N.
    InitializeP(originPolarization);
    bicgstab(BGS_MAX_ITER, BGS_MAX_ERROR);
    update_E_in_structure();
    solve_E();
}

bool DDAModel::get_HaveDevx( ) {
    return objDDAModel->Have_Devx;
}

void DDAModel::SingleResponse(int idx, bool deduction) {
    objDDAModel->SingleResponse(idx, deduction);
}

double DDAModel::GroupResponse( ) {
    return objDDAModel->GroupResponse( );
}

void DDAModel::bicgstab(int MAX_ITERATION, double MAX_ERROR){
    if (verbose) {
        cout << "--------------Calculation start. Iterative method used: BICGSTAB (" << MAX_ITERATION << " iterations)---------------" << endl;
        cout << endl;
    }
    
    high_resolution_clock::time_point t_start = high_resolution_clock::now();

    //fftw_init_threads();                                                       //////////Initialize the multi-thread
    //fftw_plan_with_nthreads(NUM_THREADS);
    //cout<<"Threads"<<NUM_THREADS<<endl;;
    VectorXcd p=VectorXcd::Zero(N*3); 
    VectorXcd t = VectorXcd::Zero(N*3);
    VectorXcd w = VectorXcd::Zero(N*3);
    VectorXcd r = VectorXcd::Zero(N*3);
    VectorXcd r0 = VectorXcd::Zero(N*3);
    VectorXcd rl = VectorXcd::Zero(N*3);
    VectorXcd y = VectorXcd::Zero(N*3);
    VectorXcd u = VectorXcd::Zero(N*3);
    VectorXcd z = VectorXcd::Zero(N*3);
    VectorXcd x = VectorXcd::Zero(N*3);
    std::complex<double> alpha;
    std::complex<double> beta;
    std::complex<double> eta;
    std::complex<double> zeta;

    VectorXcd Ax0 = Aproductwithalb(P);
    
    r = E-Ax0;
    r0 = r;
    p = r;

    VectorXcd Ap0 = Aproductwithalb(p);
    alpha = r0.dot(r)/r0.dot(Ap0);
    t = r-alpha*Ap0;

    VectorXcd At0 = Aproductwithalb(t);
    zeta = At0.dot(t)/At0.dot(At0);
    u = zeta*Ap0;
    z = zeta*r-alpha*u;
    P = P+alpha*p+z;                                    //this will directly change P in this.
    rl = r;
    r = t-zeta*At0;

    VectorXcd Ap = VectorXcd::Zero(N*3);
    VectorXcd At = VectorXcd::Zero(N*3);
    for (int it=0;it<=MAX_ITERATION-1;it++) {
        if (verbose && (it+1)%1000==0) {
            //cout << "r.norm(): " << r.norm() << endl;
            //cout << "E.norm(): " << E.norm() << endl;
            cout << "                Iter " << it+1 << ", error=" << r.norm()/E.norm() << " MAX_ERROR="<<MAX_ERROR<<endl;
        }

        complex<double> r0dotrl = r0.dot(rl);
        if (r0dotrl.real() == 0 && r0dotrl.imag() == 0) {
            Error = r.norm() / E.norm();
            //cout << "r.norm(): " << r.norm() << endl;
            //cout << "E.norm(): " << E.norm() << endl;
            ITERATION = it + 1;
            if (verbose) {
                high_resolution_clock::time_point t_end = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(t_end - t_start).count();
                time = duration_cast<milliseconds>(t_end - t_start).count();
                cout << "----------------------------r0dotrl==(0,0), nan is going to occur so stop now------------------------" << endl;
                cout << "--------------Calculation finished. Duration: " << duration / 1000.0 << "s.-------------------" << endl;

                cout << "              Error: " << Error << endl;
                cout << "              Iteration: " << ITERATION << endl;
                cout << endl;
            }
            return;
        }

        beta = (alpha/zeta)*r0.dot(r)/r0.dot(rl);
        p = r+beta*(p-u);
        Ap = Aproductwithalb(p);
        alpha = r0.dot(r)/r0.dot(Ap);
        t = r-alpha*Ap;
        At = Aproductwithalb(t);

        zeta = At.dot(t)/At.dot(At);
        u = zeta*Ap;
        z = zeta*r-alpha*u;
        P = P+alpha*p+z;
        rl = r;
        r = t-zeta*At;

        if (r.norm()/E.norm()<=MAX_ERROR) {
            Error = r.norm()/E.norm();
            
            ITERATION = it+1;
            if (verbose) {
                cout << "r.norm(): " << r.norm() << endl;
                cout << "E.norm(): " << E.norm() << endl;
                high_resolution_clock::time_point t_end = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(t_end-t_start).count();
                time = duration_cast<milliseconds>(t_end-t_start).count();
                cout << "--------------Calculation finished. Duration: " << duration/1000.0 << "s.-------------------" << endl;

                cout << "              Error: "<<Error<<endl;
                cout << "              Iteration: "<<ITERATION<<endl;
                cout << endl;
            }
            return;
        }
    }
    high_resolution_clock::time_point t_end = high_resolution_clock::now();
    time = duration_cast<milliseconds>(t_end-t_start).count();
    cout<<"                ERROR:does not converge in "<<MAX_ITERATION<<" iterations"<<endl;
    return;
}

void DDAModel::bicgstab(int MAX_ITERATION, double MAX_ERROR, int EVOITERATION) {
    if (verbose) {
        cout << "--------------Calculation start. Iterative method used: BICGSTAB---------------" << endl;
        cout << endl;
    }
    //int N = (*Core).get_N();

    high_resolution_clock::time_point t_start = high_resolution_clock::now();

    //fftw_init_threads();                                                       //////////Initialize the multi-thread
    //fftw_plan_with_nthreads(NUM_THREADS);
    //cout<<"Threads"<<NUM_THREADS<<endl;;
    VectorXcd p = VectorXcd::Zero(N * 3);
    VectorXcd t = VectorXcd::Zero(N * 3);
    VectorXcd w = VectorXcd::Zero(N * 3);
    VectorXcd r = VectorXcd::Zero(N * 3);
    VectorXcd r0 = VectorXcd::Zero(N * 3);
    VectorXcd rl = VectorXcd::Zero(N * 3);
    VectorXcd y = VectorXcd::Zero(N * 3);
    VectorXcd u = VectorXcd::Zero(N * 3);
    VectorXcd z = VectorXcd::Zero(N * 3);
    VectorXcd x = VectorXcd::Zero(N * 3);
    std::complex<double> alpha;
    std::complex<double> beta;
    std::complex<double> eta;
    std::complex<double> zeta;

    //Always starts with P=0 to avoid strange behaviour
    //P = VectorXcd::Zero(N * 3);

    VectorXcd Ax0 = Aproductwithalb(P);
    r = E - Ax0;
    r0 = r;
    p = r;
    VectorXcd Ap0 = Aproductwithalb(p);
    alpha = r0.dot(r) / r0.dot(Ap0);
    t = r - alpha * Ap0;
    VectorXcd At0 = Aproductwithalb(t);
    zeta = At0.dot(t) / At0.dot(At0);
    u = zeta * Ap0;
    z = zeta * r - alpha * u;
    P = P + alpha * p + z;                                    //this will directly change P in this.
    rl = r;
    r = t - zeta * At0;

    VectorXcd Ap = VectorXcd::Zero(N * 3);
    VectorXcd At = VectorXcd::Zero(N * 3);
    for (int it = 0; it <= MAX_ITERATION - 1; it++) {
        if (verbose && (it + 1) % 1000 == 0) {
            cout << "r.norm(): " << r.norm() << endl;
            cout << "E.norm(): " << E.norm() << endl;
            cout << "                Iter " << it + 1 << ", error=" << r.norm() / E.norm() << " MAX_ERROR=" << MAX_ERROR << endl;
        }
        beta = (alpha / zeta) * r0.dot(r) / r0.dot(rl);

        p = r + beta * (p - u);
        Ap = Aproductwithalb(p);
        alpha = r0.dot(r) / r0.dot(Ap);
        t = r - alpha * Ap;
        At = Aproductwithalb(t);

        zeta = At.dot(t) / At.dot(At);
        u = zeta * Ap;
        z = zeta * r - alpha * u;
        P = P + alpha * p + z;
        rl = r;
        r = t - zeta * At;

        if (r.norm() / E.norm() <= MAX_ERROR) {
            Error = r.norm() / E.norm();
            
            ITERATION = it + 1;
            if (verbose) {
                cout << "r.norm(): " << r.norm() << endl;
                cout << "E.norm(): " << E.norm() << endl;
                high_resolution_clock::time_point t_end = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(t_end - t_start).count();
                time = duration_cast<milliseconds>(t_end - t_start).count();
                cout << "--------------Calculation finished. Duration: " << duration / 1000.0 << "s.-------------------" << endl;
                //ofstream fout;
                //fout.open("DDATime.txt", fstream::app);
                //fout<<N<<" "<<duration/1000.0<<endl;
                //fout << ITERATION << " " << duration / 1000.0 / ITERATION << endl;
                //fout.close();

                cout << "              Error: " << Error << endl;
                cout << "              Iteration: " << ITERATION << endl;
                cout << endl;
            }
            return;
        }
    }
    high_resolution_clock::time_point t_end = high_resolution_clock::now();
    time = duration_cast<milliseconds>(t_end - t_start).count();
    cout << "                ERROR: AProductCore did not converge in " << MAX_ITERATION << " iterations. Error was " << Error << endl;
    return;
}

void DDAModel::change_E(VectorXcd E_){
    E=E_;
}

void DDAModel::reset_E(){

    for (int i=0;i<N;i++) {
        E(3 * i) = E0 * n_E0(0) * (cos(K * d * (n_K(0) * (*geometry)(3 * i) + n_K(1) * (*geometry)(3 * i + 1) + n_K(2) * (*geometry)(3 * i + 2))) + sin(K * d * (n_K(0) * (*geometry)(3 * i) + n_K(1) * (*geometry)(3 * i + 1) + n_K(2) * (*geometry)(3 * i + 2))) * 1i);
        E(3 * i + 1) = E0 * n_E0(1) * (cos(K * d * (n_K(0) * (*geometry)(3 * i) + n_K(1) * (*geometry)(3 * i + 1) + n_K(2) * (*geometry)(3 * i + 2))) + sin(K * d * (n_K(0) * (*geometry)(3 * i) + n_K(1) * (*geometry)(3 * i + 1) + n_K(2) * (*geometry)(3 * i + 2))) * 1i);
        E(3 * i + 2) = E0 * n_E0(2) * (cos(K * d * (n_K(0) * (*geometry)(3 * i) + n_K(1) * (*geometry)(3 * i + 1) + n_K(2) * (*geometry)(3 * i + 2))) + sin(K * d * (n_K(0) * (*geometry)(3 * i) + n_K(1) * (*geometry)(3 * i + 1) + n_K(2) * (*geometry)(3 * i + 2))) * 1i);
    }
}

void DDAModel::UpdateAlpha() { 

    for (int i = 0; i <= dielectric_old.size() - 1; i++) {
        int labelfloor = int(floor(dielectric_old(i)));
        int labelnext = labelfloor + 1;
        if (labelfloor >= 1) {
            labelnext = labelfloor;
        }
        std::complex<double> diel_tmp = (*material)(labelfloor) + (dielectric_old(i) - double(labelfloor)) * ((*material)(labelnext) - (*material)(labelfloor));
        diel(i) = diel_tmp;
        al(i) = 1.0 / Get_Alpha(lam, K, d, diel_tmp, n_E0, n_K);
    }
}

void DDAModel::UpdateAlphaSingle(int idx) {

    int labelfloor = int(floor(dielectric_old(3 * idx)));
    int labelnext = labelfloor + 1;
    if (labelfloor >= 1) {
        labelnext = labelfloor;
    }
    std::complex<double> diel_tmp = (*material)(labelfloor) + (dielectric_old(3 * idx) - double(labelfloor)) * ((*material)(labelnext) - (*material)(labelfloor));
    diel(3 * idx) = diel_tmp;
    diel(3 * idx + 1) = diel_tmp;
    diel(3 * idx + 2) = diel_tmp;
    al(3 * idx) = 1.0 / Get_Alpha(lam, K, d, diel_tmp, n_E0, n_K);
    al(3 * idx + 1) = al(3 * idx);
    al(3 * idx + 2) = al(3 * idx);

}

void DDAModel::solve_E(){
    if(RResultSwitch == true){

        for (int j = 0;j<int(round(RResult.size()/3));j++){
            double x = d*RResult(3*j);
            double y = d*RResult(3*j+1);
            double z = d*RResult(3*j+2);
            Vector3cd sum=Vector3cd::Zero();
            Vector3cd E_ext=Vector3cd::Zero();
            E_ext(0) = E0*n_E0(0)*(cos(K*(n_K(0)*x+n_K(1)*y+n_K(2)*z))+sin(K*(n_K(0)*x+n_K(1)*y+n_K(2)*z))*1i);
            E_ext(1) = E0*n_E0(1)*(cos(K*(n_K(0)*y+n_K(1)*y+n_K(2)*z))+sin(K*(n_K(0)*x+n_K(1)*y+n_K(2)*z))*1i);
            E_ext(2) = E0*n_E0(2)*(cos(K*(n_K(0)*z+n_K(1)*y+n_K(2)*z))+sin(K*(n_K(0)*x+n_K(1)*y+n_K(2)*z))*1i);
            for (int i=0;i<N;i++){
                double rx = x - d * (*geometry)(3 * i);                  //R has no d in it, so needs to time d
                double ry = y - d * (*geometry)(3 * i + 1);
                double rz = z - d * (*geometry)(3 * i + 2);
                Matrix3cd A=(*Core).A_dic_generator(rx,ry,rz);
                sum(0)+=(A(0,0)*P(3*i)+A(0,1)*P(3*i+1)+A(0,2)*P(3*i+2));
                sum(1)+=(A(1,0)*P(3*i)+A(1,1)*P(3*i+1)+A(1,2)*P(3*i+2));
                sum(2)+=(A(2,0)*P(3*i)+A(2,1)*P(3*i+1)+A(2,2)*P(3*i+2));
            }
            EResult(3*j) = E_ext(0)-sum(0);
            EResult(3*j+1) = E_ext(1)-sum(1);
            EResult(3*j+2) = E_ext(2)-sum(2);
        }
    }

    else
        EResult = Einternal;
  
}

void DDAModel::update_E_in_structure(){
    
    for(int i=0;i<=3*N-1;i++){
        std::complex<double> lorentzfactor = 2.0 + diel(i);
        lorentzfactor = lorentzfactor / 3.0;
        Einternal(i) = al(i) * P(i)/lorentzfactor;
    }
}

VectorXcd DDAModel::Aproductwithalb(VectorXcd& b) {
    if (b.size() != al.size()) {
        cout << "In Aproductwithalb, bsize not equal to alsize" << endl;
    }
    VectorXcd result = VectorXcd::Zero(b.size());
    for (int i = 0; i <= al.size() - 1; i++) {
        result(i) = b(i) * al(i);
    }
    return (*Core).Aproduct(b, geometry) + result;
}

void DDAModel::output_to_file(string save_position, int iteration, int ModelLabel){
    
    string name;
    name = save_position + "Model_results" + to_string(ModelLabel) + "it" + to_string(iteration) + ".txt";
    //name = save_position + "Model_output_verify\\" + "Model_results" + "it" + to_string(iteration) + ".txt";
    ofstream fout(name);

    for (int i = 0; i <= EResult.size() - 1; i++) {

        if (EResult(i).imag() < 0) {
            fout << EResult(i).real() << EResult(i).imag() << "j" << endl;
        }
        else {
            fout << EResult(i).real() << "+" << EResult(i).imag() << "j" << endl;
        }
    }

    for (int i = 0; i <= P.size() - 1; i++) {

        if (P(i).imag() < 0) {
            fout << P(i).real() << P(i).imag() << "j" << endl;
        }
        else {
            fout << P(i).real() << "+" << P(i).imag() << "j" << endl;
        }
    }

    fout.close();
}

void DDAModel::output_to_file(string save_position, int iteration) {

    string name;
    name = save_position + "Model_results" + "it" + to_string(iteration) + ".txt";
    //name = save_position + "Model_output_verify\\" + "Model_results" + "it" + to_string(iteration) + ".txt";
    ofstream fout(name);
    for (int i = 0; i <= EResult.size() - 1; i++) {

        if (EResult(i).imag() < 0) {
            fout << EResult(i).real() << EResult(i).imag() << "j" << endl;
        }
        else {
            fout << EResult(i).real() << "+" << EResult(i).imag() << "j" << endl;
        }
    }
    
    fout.close();
}

double DDAModel::get_beta( ) {
    return Filterstats->get_beta( );
}

double DDAModel::get_ita( ) {
    return Filterstats->get_ita();
}

void DDAModel::update_beta(const int iteration, const int Max_iteration) {
    Filterstats->update_beta(iteration, Max_iteration );
}

void DDAModel::InitializeP(VectorXcd& Initializer) {
    P = Initializer;
}
VectorXcd* DDAModel::get_P() {
    return &P;
}
Vector3d DDAModel::get_nE0() {
    return n_E0;
}
Vector3d DDAModel::get_nK() {
    return n_K;
}
double DDAModel::get_E0() {
    return E0;
}
VectorXcd* DDAModel::get_Einternal() {
    return &Einternal;
}
AProductCore* DDAModel::get_Core() {
    return Core;
}

VectorXcd* DDAModel::get_al() {
    return &al;
}

VectorXcd* DDAModel::get_al_max() {
    return &al_max;
}

VectorXcd* DDAModel::get_P_max() {
    return &P_max;
}

int DDAModel::get_ITERATION() {
    return ITERATION;
}

//----------------------------------------heritage from AProductCore------------------------------------

int DDAModel::get_N() {
    return N;
}
int DDAModel::get_Nx( ) {
    return Nx;
}
int DDAModel::get_Ny( ) {
    return Ny;
}
int DDAModel::get_Nz( ) {
    return Nz;
}
double DDAModel::get_lam( ) {
    return lam;
}
double DDAModel::get_d( ) {
    return d;
}
VectorXi* DDAModel::get_geometry( ) {
    return geometry;
}

VectorXi* DDAModel::get_geometryPara( ) {
    return &geometryPara;
}

VectorXd* DDAModel::get_parameters( ) {
    return &parameters;
}

// Making this return a copy, temporarily, to get python bindings to work.
VectorXd DDAModel::get_parameter_copy( ) {
    return parameters;
}

// Making this return a copy, temporarily, to get python bindings to work.
VectorXd DDAModel::get_dielectrics_copy( ) {
    return dielectric_old;
}

VectorXd* DDAModel::get_Para_origin( ) {
    if ( !Filter ) {
        cout << "ERROR: SpacePara::get_Para_origin()--Filter can not be false" << endl;
        throw 1;
    }
    return &Para_origin;
}

VectorXd* DDAModel::get_Para_filtered( ) {
    if ( !Filter ) {
        cout << "ERROR: SpacePara::get_Para_filtered()--Filter can not be false" << endl;
        throw 1;
    }
    return &Para_filtered;
}

bool DDAModel::get_Filter( ) {
    return Filter;
}

FilterOption* DDAModel::get_Filterstats( ) {
    if ( !Filter ) {
        cout << "ERROR: SpacePara::get_Filterstats()--Filter can not be false" << endl;
        throw 1;
    }
    return Filterstats;
}

vector<vector<WeightPara>>* DDAModel::get_FreeWeight( ) {
    if ( !Filter ) {
        cout << "ERROR: SpacePara::get_FreeWeight()---Filter can not be false" << endl;
        throw 1;
    }
    return &FreeWeight;
}

vector<vector<int>>* DDAModel::get_Paratogeometry( ) {
    return &Paratogeometry;
}

VectorXd* DDAModel::get_dielectric_old( ) {
    return &dielectric_old;
}
VectorXd* DDAModel::get_diel_old_max( ) {
    return &diel_old_max;
}