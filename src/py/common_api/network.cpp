/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/common_api/bindings.hpp>
#include <migraphx/common_api/utils.hpp>
#include "../common_api/include/MgxInfer.hpp"

namespace mgxinfer1 {
namespace pybinds {

using namespace py::literals;

void ilayer(py::module&);
void layers(py::module&);

Weights optional_weights(Weights*);

void network_bindings(py::module& m)
{
    ilayer(m);
    layers(m);

    py::class_<INetworkDefinition>(m, "INetworkDefinition", "TODO docstring", py::module_local())
        .def_property("name", &INetworkDefinition::getName, &INetworkDefinition::setName)
        .def_property_readonly("num_layers", &INetworkDefinition::getNbLayers)
        .def_property_readonly("num_inputs", &INetworkDefinition::getNbInputs)
        .def_property_readonly("num_outputs", &INetworkDefinition::getNbOutputs)
        .def("mark_output", &INetworkDefinition::markOutput, "tensor"_a, "TODO docstring")
        .def("add_input",
             &INetworkDefinition::addInput,
             "name"_a,
             "dtype"_a,
             "shape"_a,
             "TODO docstring",
             py::return_value_policy::reference_internal)
        .def(
            "add_convolution_nd",
            [](INetworkDefinition& self,
               ITensor& input,
               int32_t num_output_maps,
               Dims kernel_size,
               Weights kernel,
               Weights* bias) {
                return self.addConvolutionNd(
                    input, num_output_maps, kernel_size, kernel, optional_weights(bias));
            },
            "input"_a,
            "num_output_maps"_a,
            "kernel_shape"_a,
            "kernel"_a,
            "bias"_a = nullptr,
            py::keep_alive<1, 5>{},
            py::keep_alive<1, 6>{},
            "TODO docstring",
            py::return_value_policy::reference_internal)
        .def("add_activation",
             &INetworkDefinition::addActivation,
             "input"_a,
             "type"_a,
             "TODO docstring",
             py::return_value_policy::reference_internal)
        .def("add_pooling_nd",
             &INetworkDefinition::addPoolingNd,
             "input"_a,
             "type"_a,
             "window_size"_a,
             "TODO docstring",
             py::return_value_policy::reference_internal)
        .def("add_shuffle",
             &INetworkDefinition::addShuffle,
             "input"_a,
             "TODO docstring",
             py::return_value_policy::reference_internal)
        .def("add_matrix_multiply",
             &INetworkDefinition::addMatrixMultiply,
             "input0"_a,
             "op0"_a,
             "input1"_a,
             "op1"_a,
             "TODO docstring",
             py::return_value_policy::reference_internal)
        .def("add_constant",
             &INetworkDefinition::addConstant,
             "shape"_a,
             "weights"_a,
             py::keep_alive<1, 3>{},
             INetworkDefinitionDoc::add_constant,
             py::return_value_policy::reference_internal);
}

void ilayer(py::module& m)
{
    py::class_<ILayer, std::unique_ptr<ILayer>>(m, "ILayer", "TODO docstring", py::module_local())
        .def_property("name", &ILayer::getName, &ILayer::setName)
        .def_property("metadata", &ILayer::getMetadata, &ILayer::setMetadata)
        .def_property_readonly("type", &ILayer::getType)
        .def_property_readonly("num_inputs", &ILayer::getNbInputs)
        .def_property_readonly("num_outputs", &ILayer::getNbOutputs)
        .def_property("precision", &ILayer::getPrecision, &ILayer::setPrecision)
        .def_property_readonly("precision_is_set", &ILayer::precisionIsSet)
        .def("set_input", &ILayer::setInput, "index"_a, "tensor"_a, ILayerDoc::set_input)
        .def("get_input", &ILayer::getInput, "index"_a, "TODO docstring")
        .def("get_output", &ILayer::getOutput, "index"_a, "TODO docstring")
        .def("reset_precision", &ILayer::resetPrecision, "TODO docstring")
        .def("set_output_type", &ILayer::setOutputType, "index"_a, "dtype"_a, "TODO docstring")
        .def("get_output_type", &ILayer::getOutputType, "index"_a, "TODO docstring")
        .def("output_type_is_set", &ILayer::outputTypeIsSet, "index"_a, "TODO docstring")
        .def("reset_output_type", &ILayer::resetOutputType, "index"_a, "TODO docstring")
}

void layers(py::module& m)
{
    /*IActivationLayer*/
    py::enum_<ActivationType>(m, "ActivationType", "TODO docstring", py::module_local())
        .value("RELU", ActivationType::kRELU, "TODO docstring")
        .value("SIGMOID", ActivationType::kSIGMOID, "TODO docstring")
        .value("TANH", ActivationType::kTANH, "TODO docstring")
        .value("LEAKY_RELU", ActivationType::kLEAKY_RELU, "TODO docstring")
        .value("ELU", ActivationType::kELU, "TODO docstring")
        .value("SELU", ActivationType::kSELU, "TODO docstring")
        .value("SOFTSIGN", ActivationType::kSOFTSIGN, "TODO docstring")
        .value("SOFTPLUS", ActivationType::kSOFTPLUS, "TODO docstring")
        .value("CLIP", ActivationType::kCLIP, "TODO docstring")
        .value("HARD_SIGMOID", ActivationType::kHARD_SIGMOID, "TODO docstring")
        .value("SCALED_TANH", ActivationType::kSCALED_TANH, "TODO docstring")
        .value("THRESHOLDED_RELU", ActivationType::kTHRESHOLDED_RELU, "TODO docstring")
        .value("GELU_ERF", ActivationType::kGELU_ERF, "TODO docstring")
        .value("GELU_TANH", ActivationType::kGELU_TANH, "TODO docstring");

    py::class_<IActivationLayer, ILayer, std::unique_ptr<IActivationLayer>>(
        m, "IActivationLayer", "TODO docstring", py::module_local())
        .def_property(
            "type", &IActivationLayer::getActivationType, &IActivationLayer::setActivationType)
        .def_property("alpha", &IActivationLayer::getAlpha, &IActivationLayer::setAlpha)
        .def_property("beta", &IActivationLayer::getBeta, &IActivationLayer::setBeta);
    /*IActivationLayer*/

    /*IConvolutionLayer*/
    py::class_<IConvolutionLayer, ILayer, std::unique_ptr<IConvolutionLayer>>(
        m, "IConvolutionLayer", "TODO docstring", py::module_local())
        .def_property("num_output_maps",
                      &IConvolutionLayer::getNbOutputMaps,
                      &IConvolutionLayer::setNbOutputMaps)
        .def_property(
            "pre_padding", &IConvolutionLayer::getPrePadding, &IConvolutionLayer::setPrePadding)
        .def_property(
            "post_padding", &IConvolutionLayer::getPostPadding, &IConvolutionLayer::setPostPadding)
        .def_property(
            "padding_mode", &IConvolutionLayer::getPaddingMode, &IConvolutionLayer::setPaddingMode)
        .def_property(
            "num_groups", &IConvolutionLayer::getNbGroups, &IConvolutionLayer::setNbGroups)
        // Return numpy arrays instead of weights.
        .def_property(
            "kernel",
            [](IConvolutionLayer& self) { return weights_to_numpy(self.getKernelWeights()); },
            py::cpp_function(&IConvolutionLayer::setKernelWeights, py::keep_alive<1, 2>{}))
        .def_property(
            "bias",
            [](IConvolutionLayer& self) { return weights_to_numpy(self.getBiasWeights()); },
            py::cpp_function(&IConvolutionLayer::setBiasWeights, py::keep_alive<1, 2>{}))
        .def_property("kernel_size_nd",
                      &IConvolutionLayer::getKernelSizeNd,
                      &IConvolutionLayer::setKernelSizeNd)
        .def_property("stride_nd", &IConvolutionLayer::getStrideNd, &IConvolutionLayer::setStrideNd)
        .def_property(
            "padding_nd", &IConvolutionLayer::getPaddingNd, &IConvolutionLayer::setPaddingNd)
        .def_property(
            "dilation_nd", &IConvolutionLayer::getDilationNd, &IConvolutionLayer::setDilationNd);
    /*IConvolutionLayer*/

    /*IPoolingLayer*/
    py::enum_<PoolingType>(m, "PoolingType", "TODO docstring", py::module_local())
        .value("MAX", PoolingType::kMAX, "TODO docstring")
        .value("AVERAGE", PoolingType::kAVERAGE, "TODO docstring")
        .value("MAX_AVERAGE_BLEND", PoolingType::kMAX_AVERAGE_BLEND, "TODO docstring");

    py::class_<IPoolingLayer, ILayer, std::unique_ptr<IPoolingLayer>>(
        m, "IPoolingLayer", "TODO docstring", py::module_local())
        .def_property("type", &IPoolingLayer::getPoolingType, &IPoolingLayer::setPoolingType)
        .def_property("pre_padding", &IPoolingLayer::getPrePadding, &IPoolingLayer::setPrePadding)
        .def_property(
            "post_padding", &IPoolingLayer::getPostPadding, &IPoolingLayer::setPostPadding)
        .def_property(
            "padding_mode", &IPoolingLayer::getPaddingMode, &IPoolingLayer::setPaddingMode)
        .def_property(
            "blend_factor", &IPoolingLayer::getBlendFactor, &IPoolingLayer::setBlendFactor)
        .def_property("average_count_excludes_padding",
                      &IPoolingLayer::getAverageCountExcludesPadding,
                      &IPoolingLayer::setAverageCountExcludesPadding)
        .def_property(
            "window_size_nd", &IPoolingLayer::getWindowSizeNd, &IPoolingLayer::setWindowSizeNd)
        .def_property("stride_nd", &IPoolingLayer::getStrideNd, &IPoolingLayer::setStrideNd)
        .def_property("padding_nd", &IPoolingLayer::getPaddingNd, &IPoolingLayer::setPaddingNd);
    /*IPoolingLayer*/

    /*IShuffleLayer*/
    py::class_<Permutation>(m, "Permutation", "TODO docstring", py::module_local())
        .def(py::init<>())
        .def(py::init([](const std::vector<int32_t>& in) {
            int32_t const maxDims{static_cast<int32_t const>(Dims::MAX_DIMS)};
            PY_ASSERT_VALUE_ERROR(in.size() <= maxDims,
                                  "Invalid input length. Max expected length is " +
                                      std::to_string(maxDims));
            Permutation* perm = new Permutation{};
            for(int32_t i = 0; i < in.size(); ++i)
                self->order[i] = in[i];
            return perm;
        }));
    // Allow for string representations (displays like a python tuple).
    // .def("__str__", lambdas::permutation_to_str)
    // .def("__repr__", lambdas::permutation_to_str)
    // Allows for iteration.
    // .def("__getitem__", lambdas::permutation_getter)
    // .def("__setitem__", lambdas::permutation_setter)
    // .def("__len__", lambdas::permutation_len);

    // Make it possible to use tuples/lists in Python in place of Permutation.
    py::implicitly_convertible<std::vector<int32_t>, Permutation>();

    py::class_<IShuffleLayer, ILayer, std::unique_ptr<IShuffleLayer>>(
        m, "IShuffleLayer", "TODO docstring", py::module_local())
        .def_property(
            "first_transpose", &IShuffleLayer::getFirstTranspose, &IShuffleLayer::setFirstTranspose)
        .def_property("reshape_dims",
                      &IShuffleLayer::getReshapeDimensions,
                      &IShuffleLayer::setReshapeDimensions)
        .def_property("second_transpose",
                      &IShuffleLayer::getSecondTranspose,
                      &IShuffleLayer::setSecondTranspose)
        .def_property("zero_is_placeholder",
                      &IShuffleLayer::getZeroIsPlaceholder,
                      &IShuffleLayer::setZeroIsPlaceholder)
        .def("set_input", &IShuffleLayer::setInput, "index"_a, "tensor"_a, "TODO docstring");
    /*IShuffleLayer*/

    /*IConstantLayer*/
    py::class_<IConstantLayer, ILayer, std::unique_ptr<IConstantLayer>>(
        m, "IConstantLayer", "TODO docstring", py::module_local())
        .def_property(
            "weights",
            [](IConstantLayer& self) { return weights_to_numpy(self.getWeights()); },
            py::cpp_function(&IConstantLayer::setWeights, py::keep_alive<1, 2>{}))
        .def_property("shape", &IConstantLayer::getDimensions, &IConstantLayer::setDimensions);
    /*IConstantLayer*/

    /*IMatrixMultiplyLayer*/
    py::enum_<MatrixOperation>(m, "MatrixOperation", "TODO docstring", py::module_local())
        .value("NONE", MatrixOperation::kNONE, "TODO docstring")
        .value("TRANSPOSE", MatrixOperation::kTRANSPOSE, "TODO docstring")
        .value("VECTOR", MatrixOperation::kVECTOR, "TODO docstring");

    py::class_<IMatrixMultiplyLayer, ILayer, std::unique_ptr<IMatrixMultiplyLayer>>(
        m, "IMatrixMultiplyLayer", "TODO docstring", py::module_local())
        .def_property(
            "op0",
            [](IMatrixMultiplyLayer& self) { return self.getOperation(0); },
            [](IMatrixMultiplyLayer& self, MatrixOperation op) { return self.setOperation(0, op); })
        .def_property(
            "op1",
            [](IMatrixMultiplyLayer& self) { return self.getOperation(1); },
            [](IMatrixMultiplyLayer& self, MatrixOperation op) {
                return self.setOperation(1, op);
            });
    /*IMatrixMultiplyLayer*/
}

Weights optional_weights(Weights* w) { return w ? *w : Weights{DataType::kFLOAT, nullptr, 0}; }

} // namespace pybinds
} // namespace mgxinfer1
