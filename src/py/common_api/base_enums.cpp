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
#include "../common_api/include/MgxInfer.hpp"

namespace mgxinfer1 {
namespace pybinds {

void base_enum_bindings(pybind11::module& m)
{
    py::enum_<DataType>(m, "DataType", "TODO docstring", py::module_local())
        .value("FLOAT", DataType::kFLOAT, "TODO docstring")
        .value("HALF", DataType::kHALF, "TODO docstring")
        .value("BF16", DataType::kBF16, "TODO dosctring")
        .value("INT8", DataType::kINT8, "TODO dosctring")
        .value("INT32", DataType::kINT32, "TODO dosctring")
        .value("INT64", DataType::kINT64, "TODO dosctring")
        .value("BOOL", DataType::kBOOL, "TODO dosctring")
        .value("UINT8", DataType::kUINT8, "TODO dosctring")
        .value("FP8", DataType::kFP8, "TODO dosctring")
        .value("INT4", DataType::kINT4, "TODO dosctring");

    m.attr("float32")  = DataType::kFLOAT;
    m.attr("float16")  = DataType::kHALF;
    m.attr("bfloat16") = DataType::kBF16;
    m.attr("int8")     = DataType::kINT8;
    m.attr("int32")    = DataType::kINT32;
    m.attr("int64")    = DataType::kINT64;
    m.attr("bool")     = DataType::kBOOL;
    m.attr("uint8")    = DataType::kUINT8;
    m.attr("fp8")      = DataType::kFP8;
    m.attr("int4")     = DataType::kINT4;

    py::enum_<TensorIOMode>(m, "TensorIOMode", "TODO docstring", py::module_local())
        .value("NONE", TensorIOMode::kNONE, "TODO docstring")
        .value("INPUT", TensorIOMode::kINPUT, "TODO docstring")
        .value("OUTPUT", TensorIOMode::kOUTPUT, "TODO docstring");

    py::enum_<ExecutionContextAllocationStrategy>(m,
                                                  "ExecutionContextAllocationStrategy",
                                                  py::arithmetic{},
                                                  "TODO docstring",
                                                  py::module_local())
        .value("STATIC", ExecutionContextAllocationStrategy::kSTATIC, "TODO docstring")
        .value("ON_PROFILE_CHANGE",
               ExecutionContextAllocationStrategy::kON_PROFILE_CHANGE,
               "TODO docstring")
        .value("USER_MANAGED", ExecutionContextAllocationStrategy::kUSER_MANAGED, "TODO docstring");

    py::enum_<MemoryPoolType>(m, "MemoryPoolType", "TODO docstring", py::module_local())
        .value("WORKSPACE", MemoryPoolType::kWORKSPACE, "TODO docstring")
        .value("DLA_MANAGED_SRAM", MemoryPoolType::kDLA_MANAGED_SRAM, "TODO docstring")
        .value("DLA_LOCAL_DRAM", MemoryPoolType::kDLA_LOCAL_DRAM, "TODO docstring")
        .value("DLA_GLOBAL_DRAM", MemoryPoolType::kDLA_GLOBAL_DRAM, "TODO docstring")
        .value("TACTIC_DRAM", MemoryPoolType::kTACTIC_DRAM, "TODO docstring")
        .value("TACTIC_SHARED_MEMORY", MemoryPoolType::kTACTIC_SHARED_MEMORY, "TODO docstring");
}

} // namespace pybinds
} // namespace mgxinfer1
