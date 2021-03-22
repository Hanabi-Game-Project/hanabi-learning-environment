#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>


// #include "../hanabi_lib/reward_shaper.cc"
#include "../hanabi_lib/reward_shaper.h"
#include "representations.h"

namespace py = pybind11;
namespace hle = hanabi_learning_env;

void wrap_reward_shaper(py::module &m){

    
    py::class_<hle::ShapingType>(m, "ShapingType")
    .def(
        py::init<
                const int, 
                const int, 
                const int, 
                const int>(), 
        py::arg("none") = 0, 
        py::arg("risky") = 1, 
        py::arg("discard_last_of_kind") = 2, 
        py::arg("conservative") = 3                
    )
    .doc() = "Class for encoding the shaping type" ; 
    
    
    py::class_<hle::RewardShapingParams>(m, "RewardShapingParams")
    .def(
        py::init<bool,
                double,
                double,
                double,
                double,
                double,
                double>(),
        py::arg("shaper") = true,
        py::arg("min_play_probability") = 0.8,
        py::arg("w_play_penalty") = 0., 
        py::arg("m_play_penalty") = 0., 
        py::arg("w_play_reward") = 0., 
        py::arg("m_play_reward") = 0., 
        py::arg("penalty_last_of_kind") = 0. 
    )
    .doc() = "Class to store the parameters to be used in Reward Shaping";

    py::class_<hle::RewardShaper>(m, "RewardShaper")
    .def(
        py::init<   double, 
                    double, 
                    double, 
                    int, 
                    hle::RewardShapingParams,
                    hle::ShapingType
                >(),
        py::arg("performance") = 0., 
        py::arg("m_play_penalty") = 0., 
        py::arg("m_play_reward") = 0., 
        py::arg("num_ranks") = -1,
        py::arg("params") = hle::RewardShapingParams(),
        py::arg("shape_type") = hle::ShapingType(0, 1, 2, 3)
    )
    .def(
        "get_performance",
        // py::overload_cast<void> (&hle::RewardShaper::),
        &hle::RewardShaper::GetPerformance,
        "Return the performance parameter."
    )
    .def(
        "performance",
        // py::overload_cast<double> (&hle::RewardShaper::Performance), 
        &hle::RewardShaper::Performance, 
        py::arg("performance"),
        "Set the rewards and penalties based on Performance"
    )
    .def(
        "shape",
        &hle::RewardShaper::Shape, 
        py::arg("observations"), 
        py::arg("moves"), 
        "Calculate the shape of the rewards based on a set of moves and observations."
    )
    .doc() = "A class to shape rewards based on teh moves and observations"
    ;
}