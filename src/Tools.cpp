#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <map>
#include <time.h>

#include "Tools.h"

vector<string> splitInputStr(string input, string delimiter) {
    /*cout << input << endl;*/
    size_t pos = 0;
    std::string token;
    vector<string> result;
    while ((pos = input.find(delimiter)) != string::npos) {
        token = input.substr(0, pos);
        input = input.substr(pos + delimiter.size());
        result.push_back(token);
        /*cout << token << endl;*/
    }
    return result;
}

pair<VectorXi, VectorXd> InputInitial(string open_position, string model_label) {
    string name1 = open_position + "Commondata.txt";
    string name2 = open_position + "CoreStructure\\CoreStructure" + model_label + ".txt";
    ofstream TotalTime;
    TotalTime.open(open_position + "TotalTime.txt");

    ifstream fin1(name1), fin2(name2);
    int Nxtmp, Nytmp, Nztmp;
    int Ntmp;
    fin1 >> Nxtmp;
    fin1 >> Nytmp;
    fin1 >> Nztmp;
    fin1 >> Ntmp;
    cout << "Input geometry size: " << Ntmp << endl;
    VectorXi geometrytmp = VectorXi::Zero(3 * Ntmp);
    for (int i = 0; i <= Ntmp - 1; i++) {
        fin1 >> geometrytmp(3 * i);
        fin1 >> geometrytmp(3 * i + 1);
        fin1 >> geometrytmp(3 * i + 2);
    }
    VectorXd dielinput = VectorXd::Zero(3 * Ntmp);
    for (int i = 0; i <= Ntmp - 1; i++) {
        fin2 >> dielinput(3 * i);
        fin2 >> dielinput(3 * i + 1);
        fin2 >> dielinput(3 * i + 2);
    }
    fin1.close();
    fin2.close();
    return pair<VectorXi, VectorXd>(geometrytmp, dielinput);
}

pair<VectorXi, VectorXd> getInputStr(string pathCommonData, string pathPara) {
    string name1 = pathCommonData;
    string name2 = pathPara;
    

    ifstream fin1(name1), fin2(name2);
    int Nxtmp, Nytmp, Nztmp;
    int Ntmp;
    fin1 >> Nxtmp;
    fin1 >> Nytmp;
    fin1 >> Nztmp;
    fin1 >> Ntmp;
    cout << "Nx is" << Nxtmp << endl;
    cout << "Ny is" << Nytmp << endl;
    cout << "Nz is" << Nztmp << endl;
    cout << "Input geometry size: " << Ntmp << endl;
    VectorXi geometrytmp = VectorXi::Zero(3 * Ntmp);
    for (int i = 0; i <= Ntmp - 1; i++) {
        fin1 >> geometrytmp(3 * i);
        fin1 >> geometrytmp(3 * i + 1);
        fin1 >> geometrytmp(3 * i + 2);
    }
    VectorXd dielinput = VectorXd::Zero(3 * Ntmp);
    for (int i = 0; i <= Ntmp - 1; i++) {
        fin2 >> dielinput(3 * i);
        fin2 >> dielinput(3 * i + 1);
        fin2 >> dielinput(3 * i + 2);
    }
    fin1.close();
    fin2.close();
    return pair<VectorXi, VectorXd>(geometrytmp, dielinput);
}

tuple<int, int, int, int, VectorXi, VectorXd> getInputs(string pathCommonData, string pathPara) {
    string name1 = pathCommonData;
    string name2 = pathPara;


    ifstream fin1(name1), fin2(name2);
    int Nx, Ny, Nz;
    int N;
    fin1 >> Nx;
    fin1 >> Ny;
    fin1 >> Nz;
    fin1 >> N;
    cout << "Nx is" << Nx << endl;
    cout << "Ny is" << Ny << endl;
    cout << "Nz is" << Nz << endl;
    cout << "Input geometry size: " << N << endl;
    VectorXi geometrytmp = VectorXi::Zero(3 * N);
    for (int i = 0; i <= N - 1; i++) {
        fin1 >> geometrytmp(3 * i);
        fin1 >> geometrytmp(3 * i + 1);
        fin1 >> geometrytmp(3 * i + 2);
    }
    VectorXd dielinput = VectorXd::Zero(3 * N);
    for (int i = 0; i <= N - 1; i++) {
        fin2 >> dielinput(3 * i);
        fin2 >> dielinput(3 * i + 1);
        fin2 >> dielinput(3 * i + 2);
    }
    fin1.close();
    fin2.close();
    return tuple<int, int, int, int, VectorXi, VectorXd>(Nx, Ny, Nz, N, geometrytmp, dielinput);
}

tuple<int, int, int> getInputNs(string pathCommonData) {
    string name1 = pathCommonData;


    ifstream fin1(name1);
    int Nxtmp, Nytmp, Nztmp;
    int Ntmp;
    fin1 >> Nxtmp;
    fin1 >> Nytmp;
    fin1 >> Nztmp;
    fin1 >> Ntmp;
    
    return tuple<int, int, int>(Nxtmp, Nytmp, Nztmp);
}

//Find the Max and Min of the input geometry in each direction
MatrixXi find_scope_3_dim(VectorXi* x) {
    int N = round(int((*x).size()) / 3);
    MatrixXi result(3, 2);
    result(0, 0) = (*x)(0);
    result(0, 1) = (*x)(0);
    result(1, 0) = (*x)(1);
    result(1, 1) = (*x)(1);
    result(2, 0) = (*x)(2);
    result(2, 1) = (*x)(2);

    for (int i = 0; i <= N - 1; i++) {
        if (result(0, 0) >= (*x)(3 * i)) {
            result(0, 0) = (*x)(3 * i);
        }
        if (result(0, 1) <= (*x)(3 * i)) {
            result(0, 1) = (*x)(3 * i);
        }
        if (result(1, 0) >= (*x)(3 * i + 1)) {
            result(1, 0) = (*x)(3 * i + 1);
        }
        if (result(1, 1) <= (*x)(3 * i + 1)) {
            result(1, 1) = (*x)(3 * i + 1);
        }
        if (result(2, 0) >= (*x)(3 * i + 2)) {
            result(2, 0) = (*x)(3 * i + 2);
        }
        if (result(2, 1) <= (*x)(3 * i + 2)) {
            result(2, 1) = (*x)(3 * i + 2);
        }
    }
    return result;
}

VectorXd initial_diel_func(string initial_diel, int N) {
    VectorXd diel;
    if (initial_diel.compare("ZEROS") == 0) {
        diel = VectorXd::Zero(N);
    }
    else if (initial_diel.compare("0.5") == 0) {
        diel = VectorXd::Ones(N);
        diel = 0.5 * diel;
    }
    else if (initial_diel.compare("ONES") == 0) {
        diel = VectorXd::Ones(N);
    }
    else if (initial_diel.compare("RANDOM") == 0) {
        diel = VectorXd::Zero(N);
        srand(time(0));
        for (int i = 0; i <= N - 1; i++) {
            double r = ((double)rand() / (RAND_MAX));
            diel(i) = r;
        }

    }
    else {
        diel = VectorXd::Zero(N);
        cout << "ERROR : VectorXd initial_diel_func(string initial_diel, int N) : The initial type given does not match any of the built in method" << endl;
        throw 1;
    }
    return diel;
}

double initial_diel_func(string initial_diel) {
    VectorXd diel;
    if (initial_diel.compare("ZEROS") == 0) {
        return 0.0;
    }
    else if (initial_diel.compare("0.5") == 0) {
        return 0.5;
    }
    else if (initial_diel.compare("ONES") == 0) {
        return 1.0;
    }
    else if (initial_diel.compare("RANDOM") == 0) {
        return ((double)rand() / (RAND_MAX));

    }
    else {
        cout << "ERROR : double initial_diel_func(string initial_diel) : The initial type given does not match any of the built in method" << endl;
        throw 1;
    }
}

double initial_diel_func(double initial_diel) {
    VectorXd diel;
    if (initial_diel<=1.0 && initial_diel>=0.0) {
        return initial_diel;
    }
   
    else {
        cout << "ERROR : double initial_diel_func(double initial_diel): Value given for the para not within 0.0 and 1.0" << endl;
        throw 1;
    }
}

// builds a 3*Nx*Ny*Nz vector for the position of each pixel. example: (0, 0, 0, 1, 0, 0, 2, 0, 0, ...)
VectorXi build_a_bulk(int Nx, int Ny, int Nz){
    VectorXi result=VectorXi::Zero(3*Nx*Ny*Nz);
    for(int x=0;x<=Nx-1;x++){
        for(int y=0;y<=Ny-1;y++){
            for(int z=0;z<=Nz-1;z++){
                int position=z+Nz*(y+Ny*x);
                result(3*position)=x;
                result(3*position+1)=y;
                result(3*position+2)=z;
            }
        }
    }   
    return result; 
}


complex<double> Get_material(string mat, double wl, string unit){
    map<string,double> unit_dic;
    unit_dic.insert(pair<string,double>("nm",1.0e9));
    unit_dic.insert(pair<string,double>("um",1.0e6));
    unit_dic.insert(pair<string,double>("m",1.0));
    map<string,string> diel_dic;
    diel_dic.insert(pair<string,string>("Ag","diel/Ag (Silver) - CRC (raw)"));
    diel_dic.insert(pair<string,string>("Al","diel/Al (Aluminium) - Palik (raw)"));
    diel_dic.insert(pair<string,string>("Al2O3", "diel/Al2O3(Malitson)"));
    diel_dic.insert(pair<string,string>("Au","diel/Au (Gold) - Johnson and Christy (raw)"));
    diel_dic.insert(pair<string,string>("Si","diel/Si (Silicon) - Palik (raw)"));
    diel_dic.insert(pair<string,string>("SiO2","diel/SiO2 (Glass) - Palik (raw)"));
    diel_dic.insert(pair<string,string>("TiO2", "diel/TiO2_ALD (raw)"));
    diel_dic.insert(pair<string,string>("Air","diel/Air"));
    diel_dic.insert(pair<string,string>("1.5","diel/Diel1.5"));
    diel_dic.insert(pair<string,string>("2.0","diel/Diel2.0"));
    diel_dic.insert(pair<string,string>("2.5","diel/Diel2.5"));
    diel_dic.insert(pair<string,string>("3.0","diel/Diel3.0"));
    diel_dic.insert(pair<string,string>("3.5","diel/Diel3.5"));
    diel_dic.insert(pair<string,string>("4.0","diel/Diel4.0"));
    diel_dic.insert(pair<string,string>("4.5","diel/Diel4.5"));
    diel_dic.insert(pair<string,string>("5.0","diel/Diel5.0"));
    diel_dic.insert(pair<string, string>("H2O", "diel/H2O"));
    diel_dic.insert(pair<string, string>("TiN", "diel/TiN-Pfluger"));
    diel_dic.insert(pair<string, string>("Ti", "diel/Ti-JC"));
    wl=wl/unit_dic[unit];
    string real="Re_eps.txt";
    string imag="Im_eps.txt";
    string mat_real_name=diel_dic[mat]+real;
    string mat_imag_name=diel_dic[mat]+imag; 
    double mat_real,mat_imag,a,b,up,down,up_value,down_value;
    
    up=1.0;
    down=0.0;

    ifstream mat_real_file;
    mat_real_file.open(mat_real_name);
    while(mat_real_file>>a>>b){
        if(a<=wl&&a>down){
            down=a;
            down_value=b;
        }
        else if(a>wl&&a<up){
            up=a;
            up_value=b;
        }

    }
    mat_real_file.close();
    if(up==1.0||down==0.0){
        cout<<"ERROR, wavelength out of range of the diel file provided."<<endl;
    }
    else{
        mat_real=down_value+(wl-down)*(up_value-down_value)/(up-down); 
        cout << "real:" << mat_real << endl;
    }
    up=1.0;
    down=0.0;

    ifstream mat_imag_file;
    mat_imag_file.open(mat_imag_name);
    while(mat_imag_file>>a>>b){
        if(a<=wl&&a>down){
            down=a;
            down_value=b;
        }
        else if(a>wl&&a<up){
            up=a;
            up_value=b;
        }

    }
    mat_imag_file.close();
    if(up==1.0||down==0.0){
        cout<<"ERROR, wavelength out of range of the diel file provided."<<endl;
    }
    else{
        mat_imag=down_value+(wl-down)*(up_value-down_value)/(up-down); 
        cout << "imag:" << mat_imag << endl;
    }
    
    complex<double> result=mat_real+1.0i*mat_imag;
    return result;
}

Vector2cd Get_2_material(string sub, string mat, double wl, string unit){
    Vector2cd result;
    result(0)=Get_material(sub,wl,unit);
    result(1)=Get_material(mat,wl,unit);
    return result;
}

VectorXcd Get_X_material(list<string> mat_l, double wl, string unit) {
    list<string>::iterator it = mat_l.begin();
    VectorXcd result = VectorXcd::Zero(mat_l.size());
    int i = 0;
    while (it != mat_l.end()) {
        result(i) = Get_material(*it, wl, unit);
        i++;
        it++;
    }
    return result;
}

complex<double> Get_Alpha(double lam, double K, double d, complex<double> diel, Vector3d n_E0, Vector3d n_K){
    double b1 = -1.891531;
    double b2 = 0.1648469;
    double b3 = -1.7700004;
    //cout<<"lam"<<lam<<"K"<<K<<"d"<<d<<endl;
    std::complex<double> a1=(3*pow(d,3)/(4*M_PI))*(diel-1.0)/(diel+2.0);
    //cout<<a1<<endl;
    std::complex<double> result=0.0+(2.0/3.0)*pow(K*d,3)*1.0i;
    double S = pow(n_E0(0) * n_K(0), 2) + pow(n_E0(1) * n_K(1), 2) + pow(n_E0(2) * n_K(2), 2);
    result = 1.0 + (a1 / pow(d, 3)) * ((b1 + diel * b2 + diel * b3 * S) * pow(K * d, 2) - result);
    //cout<<result<<endl;
    result=a1/result;
    return result;
}

complex<double> Get_Alpha_FCD(double lam, double K, double d, complex<double> diel) {
    complex<double> M = (4.0 / 3.0) * pow((K * d), 2) + (2.0 / 3.0) * (1.0i + (1 / M_PI) * log((M_PI - K * d) / (M_PI + K * d))) * pow(K * d, 3);
    // complex<double> kappa = (diel - 1.0) / (4 * M_PI);
    // complex<double> result = pow(d, 3) * kappa / (1.0 + (4.0 * M_PI / 3.0 - M) * kappa);
    complex<double> a1 = (3 * pow(d, 3) / (4 * M_PI)) * (diel - 1.0) / (diel + 2.0);
    complex<double> result = 1.0 - M * a1 / pow(d, 3);
    result = a1 / result;
    return result;
}

bool CheckPerp(Vector3d v1, Vector3d v2) {
    if (abs(v1.dot(v2)) <= 0.0001) {
        //cout << v1.dot(v2) << endl;
        return true;
    }
    else {
        return false;
    }
}

Vector3d nEPerpinXZ(double theta, double phi) {
    double tmp = cos(theta) / sqrt(pow(sin(theta), 2) * pow(cos(phi), 2) + pow(cos(theta), 2));
    double x[2] = { -tmp,tmp };
    double z[2] = { -sqrt(1 - tmp * tmp),sqrt(1 - tmp * tmp) };
    Vector3d k, nE;
    k << sin(theta) * cos(phi), sin(theta)* sin(phi), cos(theta);
    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            nE << x[i], 0.0, z[j];
            if (CheckPerp(k, nE) == true && x[i] >= 0.0) {
                return nE;
            }
        }
    }

    cout << "ERROR : perp nE not found in Vector3d nEPerpinXZ(double theta, double phi)" << endl;

    return nE;
}

list<double> makelist(double start, double end, double interval) {
    list<double> result;
    int number = floor((end - start) / interval + 1);
    for (int i = 0; i <= number - 1; i++) {
        result.push_back(start + i * interval);
    }
    return result;
}

list<double> makelist(double start, double end, int number) {
    list<double> result;
    double interval = (end - start) / (double(number) - 1.0);
    for (int i = 0; i <= number - 1; i++) {
        //cout << start + i * interval << endl;
        result.push_back(start + i * interval);
    }
    return result;
}

list<string> ReadMat(string input) {
    vector<string> split1 = splitInputStr(input, "/");
    list<string> result;
    for (int i = 0; i < split1.size(); i++) {
        result.push_back(split1[i]);
    }
    return result;
}

vector<double> ReadLam(string input) {
    vector<string> split1 = splitInputStr(input, "/");
    vector<double> result1;
    for (int i = 0; i < split1.size(); i++) {
        result1.push_back(stod(split1[i]));
        
    }
    vector<double> result;
    if (result1.size() == 1) {
        result.push_back(result1[0]);
    }
    else if(result1.size() == 3) {
        double start = result1[0];
        double step = result1[1];
        double end = result1[2];
        if (end > start && step>0) {
            int num = int(round((end - start) / step)) + 1;
            for (int i = 0; i < num; i++) {
                result.push_back(start + i * step);
            }
        }
        else {
            cout << "ERROR : vector<double> ReadLam(string input) : for wavelength sweep, end<=start or step<=0" << endl;
            throw 1;
        }
    }
    else {
        cout << "ERROR : vector<double> ReadLam(string input) : wavelength sweep size or format not correct." << endl;
        throw 1;
    }


    return result;
}



double exp_update(const double x, const double x_max, const double y_min, const double y_max) {
    int base = 100;
    if (x <= 300) {
        return y_min + (y_max - y_min) * (pow(base, (x / x_max)) - 1) / (base - 1);
    }
    else
        return 0.5;
}

double piecewise_update(const double x, const double x_max, const double y_min, const double y_max) {
    if (x <= 0.5 * x_max) {
        return y_min;
    }
    else if(0.5 * x_max < x && x <= 0.7 * x_max) {
        return y_min + (y_max - y_min) / 5;
    }
    else if (0.7 * x_max < x && x <= 0.8 * x_max) {
        return y_min + (y_max - y_min) / 2.5;
    }
    else {
        return y_max;
    }

}

double piecewise_update_absolute(const double x, const double x_max, const double y_min, const double y_max) {
    if (x <= 100) {
        return y_min;
    }
    else if (100 < x && x <= 140) {
        return y_min + (y_max - y_min) / 5;
    }
    else if (140 < x && x <= 160) {
        return y_min + (y_max - y_min) / 2.5;
    }
    else {
        return y_max;
    }

}

double linear_update(const double x, const double x_max, const double y_min, const double y_max) {
    return y_min + (y_max - y_min) * x / x_max;
}


double calculatePenalty(VectorXd& parameters) {
    double penalty = 0.0;
   // double coeff = 10.0;
    for (int i = 0; i < parameters.size(); i++) {
        double pixel = parameters(i);
        penalty += pixel*(1 - pixel);
    }

    return penalty;
}