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

void dims(py::module&);
void host_memory(py::module&);
void weights(py::module&);
void tensor(py::module&);

void base_type_bindings(py::module& m)
{
    host_memory(m);
    dims(m);
    weights(m);
    tensor(m);
}

void dims(py::module& m)
{
    using namespace py::literals;

    constexpr auto dims_from_vec = [](std::vector<int64_t> const& in) {
        int32_t const maxDims{static_cast<int32_t>(Dims::MAX_DIMS)};
        PY_ASSERT_VALUE_ERROR(in.size() <= maxDims,
                              "Input length " + std::to_string(in.size()) +
                                  ". Max expected length is " + std::to_string(maxDims));

        Dims* dims   = new Dims{};
        dims->nbDims = in.size();
        for(int32_t i = 0; i < in.size(); ++i)
            dims->d[i] = in[i];
        return dims;
    };

    constexpr auto dims_getitem = [](const Dims& dims, const int32_t py_idx) -> const int64_t& {
        const int32_t idx{(py_idx < 0) ? static_cast<int32_t>(dims.nbDims) + py_idx : py_idx};
        PY_ASSERT_INDEX_ERROR(idx >= 0 && idx < dims.nbDims);
        return dims.d[idx];
    };

    constexpr auto dims_getitem_slice = [](const Dims& dims, py::slice slice) {
        size_t start, stop, step, slice_len;
        PY_ASSERT_VALUE_ERROR(slice.compute(dims.nbDims, &start, &stop, &step, &slice_len),
                              "Incorrect getter slice dims");
        PY_ASSERT_INDEX_ERROR(stop <= dims.nbDims);

        py::tuple ret{slice_len};
        for(int32_t i = start, idx = 0; i < stop; i += step, ++idx)
            ret[idx] = dims.d[i];
        return ret;
    };

    constexpr auto dims_to_str = [](const Dims& dims) {
        if(dims.nbDims == 0)
            return std::string("()");

        if(dims.nbDims == 1)
            return "(" + std::to_string(dims.d[0]) + ",)";

        std::string temp = "(";
        for(int32_t i = 0; i < dims.nbDims - 1; ++i)
            temp += std::to_string(dims.d[i]) + ", ";
        temp += std::to_string(dims.d[dims.nbDims - 1]) + ")";
        return temp;
    };

    constexpr auto dims_setitem = [](Dims& dims, const int32_t py_idx, const int64_t item) {
        const int32_t idx{(py_idx < 0) ? static_cast<int32_t>(dims.nbDims) + py_idx : py_idx};
        PY_ASSERT_INDEX_ERROR(idx >= 0 && idx < dims.nbDims);
        dims.d[idx] = item;
    };

    constexpr auto dims_setitem_slice = [](Dims& dims, py::slice slice, const Dims& other) {
        size_t start, stop, step, slice_len;
        PY_ASSERT_VALUE_ERROR(slice.compute(dims.nbDims, &start, &stop, &step, &slice_len),
                              "Incorrect setter slice dims");
        PY_ASSERT_INDEX_ERROR(stop < dims.nbDims);

        for(int32_t i = start, index = 0; i < stop; i += step, ++index)
            dims.d[i] = other.d[index];
    };

    py::class_<Dims>(m, "Dims", "TODO docstring", py::module_local())
        .def(py::init<>())
        .def(py::init(dims_from_vec), "shape"_a)
        .def_property_readonly_static(
            "MAX_DIMS",
            [](py::object) { return static_cast<const int32_t>(Dims::MAX_DIMS); },
            "TODO docstring")
        .def("__len__", [](const Dims& dims) { return dims.nbDims; })
        .def("__getitem__", dims_getitem)
        .def("__getitem__", dims_getitem_slice)
        .def("__setitem__", dims_setitem)
        .def("__setitem__", dims_setitem_slice)
        .def("__str__", dims_to_str)
        .def("__repr__", dims_to_str);

    py::implicitly_convertible<std::vector<int64_t>, Dims>();

    py::class_<Dims2, Dims>(m, "Dims2", "TODO docstring", py::module_local())
        .def(py::init<>())
        .def(py::init<int64_t, int64_t>(), "dim0"_a, "dim1"_a)
        .def(py::init([](const std::vector<int64_t>& in) {
                 PY_ASSERT_VALUE_ERROR(in.size() == 2,
                                       "Input length " + std::to_string(in.size()) +
                                           " not equal to expected Dims2 length, which is 2");
                 return new Dims2{in[0], in[1]};
             }),
             "shape"_a);

    py::implicitly_convertible<std::vector<int64_t>, Dims2>();

    py::class_<DimsHW, Dims2>(m, "DimsHW", "TODO docstring", py::module_local())
        .def(py::init<>())
        .def(py::init<int64_t, int64_t>(), "h"_a, "w"_a)
        .def(py::init([](const std::vector<int64_t>& in) {
                 PY_ASSERT_VALUE_ERROR(in.size() == 2,
                                       "Input length " + std::to_string(in.size()) +
                                           " not equal to expected DimsHW length, which is 2");
                 return new DimsHW{in[0], in[1]};
             }),
             "shape"_a)
        .def_property(
            "h",
            [](const DimsHW& dims) { return dims.h(); },
            [](DimsHW& dims, int64_t i) { dims.h() = i; })
        .def_property(
            "w",
            [](DimsHW const& dims) { return dims.w(); },
            [](DimsHW& dims, int64_t i) { dims.w() = i; });

    py::implicitly_convertible<std::vector<int64_t>, DimsHW>();

    py::class_<Dims3, Dims>(m, "Dims3", "TODO docstring", py::module_local())
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t>(), "dim0"_a, "dim1"_a, "dim2"_a)
        .def(py::init([](const std::vector<int64_t>& in) {
                 PY_ASSERT_VALUE_ERROR(in.size() == 3,
                                       "Input length " + std::to_string(in.size()) +
                                           " not equal to expected Dims3 length, which is 3");
                 return new Dims3{in[0], in[1], in[2]};
             }),
             "shape"_a);

    py::implicitly_convertible<std::vector<int64_t>, Dims3>();

    py::class_<Dims4, Dims>(m, "Dims4", "TODO docstring", py::module_local())
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(), "dim0"_a, "dim1"_a, "dim2"_a, "dim3"_a)
        .def(py::init([](const std::vector<int64_t>& in) {
                 PY_ASSERT_VALUE_ERROR(in.size() == 4,
                                       "Input length " + std::to_string(in.size()) +
                                           " not equal to expected Dims4 length, which is 4");
                 return new Dims4{in[0], in[1], in[2], in[3]};
             }),
             "shape"_a);

    py::implicitly_convertible<std::vector<int64_t>, Dims4>();

    // TODO make this work for any python iterable
    m.def("volume", [](const Dims& dims) {
        size_t ret = 1;
        for(auto i = 0; i < dims.nbDims; ++i)
            ret *= dims.d[i];

        return ret;
    });
}

void host_memory(py::module& m)
{
    py::class_<IHostMemory>(
        m, "IHostMemory", py::buffer_protocol(), "TODO docstring", py::module_local())
        .def_property_readonly("dtype", [](IHostMemory const& mem) { return mem.type(); })
        .def_property_readonly("nbytes", [](IHostMemory const& mem) { return mem.size(); })
        .def_buffer([](IHostMemory& mem) {
            py::buffer_info mem_info;
            mem_info.ptr      = mem.data();
            mem_info.itemsize = sizeofDataType(mem.type());
            // TODO this should be based on mem.type
            mem_info.format  = py::format_descriptor<char>::format();
            mem_info.ndim    = 1;
            mem_info.shape   = {static_cast<py::ssize_t>(mem.size())};
            mem_info.strides = {mem_info.itemsize};
            return mem_info;
        });
}

void weights(py::module& m)
{
    py::class_<Weights>(m, "Weights", "TODO docstring", py::module_local())
        .def(py::init([](const DataType& t) {
                 return new Weights{t, nullptr, 0};
             }),
             "type"_a = DataType::kFLOAT,
             "TODO docstring")
        .def(py::init([](const DataType& t, const size_t vals, int64_t count) {
                 return new Weights{t, reinterpret_cast<void*>(vals), count};
             }),
             "type"_a,
             "ptr"_a,
             "count"_a,
             "TODO docstring")
        .def(py::init([](py::array& arr) {
                 arr = py::array::ensure(arr);
                 PY_ASSERT_VALUE_ERROR(arr,
                                       "Could not convert NumPy array to Weights. Is it using a "
                                       "data type supported by TensorRT?");
                 PY_ASSERT_VALUE_ERROR((arr.flags() & py::array::c_style),
                                       "Could not convert non-contiguous NumPy array to Weights. "
                                       "Please use numpy.ascontiguousarray() to fix this.");
                 return new Weights{dtype_to_type(arr.dtype()), arr.data(), arr.size()};
             }),
             "a"_a,
             py::keep_alive<1, 2>(),
             "TODO docstring")
        .def_property_readonly("dtype", [](const Weights& self) { return self.type; })
        .def_property_readonly("size", [](const Weights& self) { return self.count; })
        .def_property_readonly(
            "nbytes", [](const Weights& self) { return sizeofDataType(self.type) * self.count; })
        .def("numpy",
             &weights_to_numpy,
             py::return_value_policy::reference_internal,
             "TODO docstring")
        .def("__len__", [](Weights const& self) { return static_cast<size_t>(self.count); });

    py::implicitly_convertible<py::array, Weights>();
}

void tensor(py::module& m)
{
    py::enum_<TensorFormat>(
        m, "TensorFormat", "TODO docstring", py::arithmetic{}, py::module_local())
        .value("LINEAR", TensorFormat::kLINEAR, "TODO docstring")
        .value("CHW2", TensorFormat::kCHW2, "TODO docstring")
        .value("HWC8", TensorFormat::kHWC8, "TODO docstring")
        .value("CHW4", TensorFormat::kCHW4, "TODO docstring")
        .value("CHW16", TensorFormat::kCHW16, "TODO docstring")
        .value("CHW32", TensorFormat::kCHW32, "TODO docstring")
        .value("DHWC8", TensorFormat::kDHWC8, "TODO docstring")
        .value("CDHW32", TensorFormat::kCDHW32, "TODO docstring")
        .value("HWC", TensorFormat::kHWC, "TODO docstring")
        .value("DLA_LINEAR", TensorFormat::kDLA_LINEAR, "TODO docstring")
        .value("DLA_HWC4", TensorFormat::kDLA_HWC4, "TODO docstring")
        .value("HWC16", TensorFormat::kHWC16, "TODO docstring")
        .value("DHWC", TensorFormat::kDHWC, "TODO docstring");

    py::class_<ITensor, std::unique_ptr<ITensor, py::nodelete>>(
        m, "ITensor", "TODO docstring", py::module_local())
        .def_property("name", &ITensor::getName, &ITensor::setName)
        .def_property("shape", &ITensor::getDimensions, &ITensor::setDimensions)
        .def_property("dtype", &ITensor::getType, &ITensor::setType)
        .def_property("location", &ITensor::getLocation, &ITensor::setLocation)
        .def_property("allowed_formats", &ITensor::getAllowedFormats, &ITensor::setAllowedFormats)
        .def_property_readonly("is_network_input", &ITensor::isNetworkInput)
        .def_property_readonly("is_network_output", &ITensor::isNetworkOutput)
        .def_property_readonly("is_shape_tensor", &ITensor::isShapeTensor)
        .def_property_readonly("is_execution_tensor", &ITensor::isExecutionTensor)
        .def_property("allowed_formats", &ITensor::getAllowedFormats, &ITensor::setAllowedFormats);
}

} // namespace pybinds
} // namespace mgxinfer1
