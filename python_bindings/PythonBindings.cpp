#include "DDAModel.h"

#include <pybind11/pybind11.h>

#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

namespace py = pybind11;


PYBIND11_MODULE(_dda_model, module) { // NOLINT

    py::class_<DDAModel, std::unique_ptr<DDAModel>>(module, "DDAModel")
        .def(py::init([](
            const double betaMin, const double betaMax, const double ita, const std::string betaType,
            const std::vector<int> filterIterations, const std::vector<double> filterRadii, const bool useFilter,
            const std::string symmetryType, const std::vector<double> symmetryAxis, const bool isPeriodic,
            const std::string objectiveName, const std::vector<double> objPara,
            Eigen::VectorXi geometry, Eigen::VectorXd inputDielectrics,
            const int num_x, const int num_y, const int num_z, const int num_total, Eigen::Vector3d n_K,
            const double E0, Eigen::Vector3d n_E0,
            const double lam, Eigen::VectorXcd material,
            const double nback, const int MAXm, const int MAXn, const double Lm, const double Ln,
            const std::string matrixAMethodName, const double d,
            Eigen::VectorXd sineIntegralValues, Eigen::VectorXd cosineIntegralValues, const double integralDelta,
            const bool verbose
        ) {
            return std::unique_ptr<DDAModel>(new DDAModel(
                betaMin, betaMax, ita, betaType,
                filterIterations, filterRadii, useFilter,
                symmetryType, symmetryAxis, isPeriodic,
                objectiveName, objPara, &geometry, &inputDielectrics,
                num_x, num_y, num_z, num_total, n_K,
                E0, n_E0,
                lam, material,
                nback, MAXm, MAXn, Lm, Ln,
                matrixAMethodName, d,
                sineIntegralValues, cosineIntegralValues, integralDelta,
                verbose
            ));
        }),
        py::arg("betaMin"), py::arg("betaMax"), py::arg("ita"), py::arg("betaType"), 
        py::arg("filterIterations"), py::arg("filterRadii"), py::arg("useFilter"), 
        py::arg("symmetryType"), py::arg("symmetryAxis"), py::arg("isPeriodic"), 
        py::arg("objectiveName"), py::arg("objPara"), 
        py::arg("geometry"), py::arg("inputDielectrics"), 
        py::arg("num_x"), py::arg("num_y"), py::arg("num_z"), py::arg("num_total"), py::arg("n_K"), 
        py::arg("E0"), py::arg("n_E0"), 
        py::arg("lam"), py::arg("material"), 
        py::arg("nback"), py::arg("MAXm"), py::arg("MAXn"), py::arg("Lm"), py::arg("Ln"), 
        py::arg("matrixAMethodName"), py::arg("d"),
        py::arg("sineIntegralValues"), py::arg("cosineIntegralValues"), py::arg("integralDelta"),
        py::arg("verbose")
        )
        // Functions to calculate gradient and objective.
        .def("calculateObjective", &DDAModel::calculateObjective)
        .def("calculateGradients", &DDAModel::calculateGradients)
        // Function to re-calculate internal state (must be called after parameter update).
        .def("solveElectricField", &DDAModel::solveElectricField)
        // Getters and setters for the parameters.
        .def("setParameters", &DDAModel::UpdateParameters)
        .def("getParameters", &DDAModel::get_parameter_copy)
        .def("getDielectrics", &DDAModel::get_dielectrics_copy);
}
