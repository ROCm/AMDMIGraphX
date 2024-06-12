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

} // namespace pybinds
} // namespace mgxinfer1