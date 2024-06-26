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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../../../common_api/include/MgxInfer.hpp"

namespace mgxinfer1 {
namespace pybinds {

inline void throwPyError(PyObject* type, std::string const& message)
{
    PyErr_SetString(type, message.data());
    throw pybind11::error_already_set();
}

#define PY_ASSERT_RUNTIME_ERROR(assertion, msg)    \
    do                                             \
    {                                              \
        if(!(assertion))                           \
        {                                          \
            throwPyError(PyExc_RuntimeError, msg); \
        }                                          \
    } while(false)

#define PY_ASSERT_INDEX_ERROR(assertion)                     \
    do                                                       \
    {                                                        \
        if(!(assertion))                                     \
        {                                                    \
            throwPyError(PyExc_IndexError, "Out of bounds"); \
        }                                                    \
    } while(false)

#define PY_ASSERT_VALUE_ERROR(assertion, msg)    \
    do                                           \
    {                                            \
        if(!(assertion))                         \
        {                                        \
            throwPyError(PyExc_ValueError, msg); \
        }                                        \
    } while(false)

#define PY_ASSERT_TYPE_ERROR(assertion, msg)    \
    do                                          \
    {                                           \
        if(!(assertion))                        \
        {                                       \
            throwPyError(PyExc_TypeError, msg); \
        }                                       \
    } while(false)

inline DataType dtype_to_type(pybind11::dtype const& type)
{
    if(type.is(pybind11::dtype("f4")))
    {
        return mgxinfer1::DataType::kFLOAT;
    }
    else if(type.is(pybind11::dtype("f2")))
    {
        return mgxinfer1::DataType::kHALF;
    }
    else if(type.is(pybind11::dtype("i8")))
    {
        return mgxinfer1::DataType::kINT64;
    }
    else if(type.is(pybind11::dtype("i4")))
    {
        return mgxinfer1::DataType::kINT32;
    }
    else if(type.is(pybind11::dtype("i1")))
    {
        return mgxinfer1::DataType::kINT8;
    }
    else if(type.is(pybind11::dtype("b1")))
    {
        return mgxinfer1::DataType::kBOOL;
    }
    else if(type.is(pybind11::dtype("u1")))
    {
        return mgxinfer1::DataType::kUINT8;
    }

    constexpr int32_t kBITS_PER_BYTE{8};
    std::stringstream ss{};
    ss << "[E] Could not implicitly convert NumPy data type: " << type.kind()
       << (type.itemsize() * kBITS_PER_BYTE) << " to TensorRT.";
    std::cerr << ss.str() << std::endl;
    PY_ASSERT_VALUE_ERROR(false, ss.str());
    return mgxinfer1::DataType::kFLOAT;
}

inline std::unique_ptr<py::dtype> type_to_dtype(DataType type)
{
    constexpr auto make_dtype = [](char const* type_str) {
        return std::make_unique<pybind11::dtype>(type_str);
    };

    switch(type)
    {
    case mgxinfer1::DataType::kFLOAT: return make_dtype("f4");
    case mgxinfer1::DataType::kHALF: return make_dtype("f2");
    case mgxinfer1::DataType::kINT8: return make_dtype("i1");
    case mgxinfer1::DataType::kINT32: return make_dtype("i4");
    case mgxinfer1::DataType::kINT64: return make_dtype("i8");
    case mgxinfer1::DataType::kBOOL: return make_dtype("b1");
    case mgxinfer1::DataType::kUINT8: return make_dtype("u1");
    default: return nullptr;
    }
}

inline pybind11::object weights_to_numpy(const mgxinfer1::Weights& self)
{
    const auto dtype = type_to_dtype(self.type);
    if(dtype)
    {
        return pybind11::array(*dtype, self.count, self.values, pybind11::cast(self));
    }
    return pybind11::cast(self);
}

} // namespace pybinds
} // namespace mgxinfer1
