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

// https://pybind11.readthedocs.io/en/stable/advanced/classes.html
class PyLoggerTrampoline : public ILogger
{
    public:
    virtual void log(Severity severity, const char* msg) noexcept override
    {
        // TODO
        PYBIND11_OVERRIDE_PURE_NAME(void, ILogger, "log", log, severity, msg);
    }
};

class PythonLogger : public ILogger
{
    public:
    PythonLogger(Severity min = Severity::kWARNING) {}

    void log(Severity, AsciiChar const*) noexcept override {}
};

void logger_bindings(py::module& m)
{
    using namespace py::literals;

    auto ilogger =
        py::class_<ILogger, PyLoggerTrampoline>(m, "ILogger", "TODO dosctring", py::module_local())
            .def(py::init<>())
            .def("log", &ILogger::log, "severity"_a, "msg"_a, "TODO dosctring");

    py::enum_<ILogger::Severity>(
        ilogger, "Severity", py::arithmetic(), "TODO dosctring", py::module_local())
        .value("INTERNAL_ERROR", ILogger::Severity::kINTERNAL_ERROR, "TODO dosctring")
        .value("ERROR", ILogger::Severity::kERROR, "TODO dosctring")
        .value("WARNING", ILogger::Severity::kWARNING, "TODO dosctring")
        .value("INFO", ILogger::Severity::kINFO, "TODO dosctring")
        .value("VERBOSE", ILogger::Severity::kVERBOSE, "TODO dosctring")
        .export_values();

    py::class_<PythonLogger, ILogger>(m, "Logger", "TODO dosctring", py::module_local())
        .def(py::init<ILogger::Severity>(), "min_severity"_a = ILogger::Severity::kWARNING)
        .def("log", &PythonLogger::log, "severity"_a, "msg"_a, "TODO dosctring");
}

} // namespace pybinds
} // namespace mgxinfer1
