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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <migraphx/program.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/autocast_fp8.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/json.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/float8.hpp>
#include <migraphx/pass_manager.hpp>
#include "../common_api/include/MgxInfer.hpp"
#ifdef HAVE_GPU
#include <migraphx/gpu/hip.hpp>
#endif

using half   = half_float::half;
namespace py = pybind11;

#ifdef __clang__
#define MIGRAPHX_PUSH_UNUSED_WARNING \
    _Pragma("clang diagnostic push") \
        _Pragma("clang diagnostic ignored \"-Wused-but-marked-unused\"")
#define MIGRAPHX_POP_WARNING _Pragma("clang diagnostic pop")
#else
#define MIGRAPHX_PUSH_UNUSED_WARNING
#define MIGRAPHX_POP_WARNING
#endif
#define MIGRAPHX_PYBIND11_MODULE(...) \
    MIGRAPHX_PUSH_UNUSED_WARNING      \
    PYBIND11_MODULE(__VA_ARGS__)      \
    MIGRAPHX_POP_WARNING

#define MIGRAPHX_PYTHON_GENERATE_SHAPE_ENUM(x, t) .value(#x, migraphx::shape::type_t::x)
namespace migraphx {

migraphx::value to_value(py::kwargs kwargs);
migraphx::value to_value(py::list lst);

template <class T, class F>
void visit_py(T x, F f)
{
    if(py::isinstance<py::kwargs>(x))
    {
        f(to_value(x.template cast<py::kwargs>()));
    }
    else if(py::isinstance<py::list>(x))
    {
        f(to_value(x.template cast<py::list>()));
    }
    else if(py::isinstance<py::bool_>(x))
    {
        f(x.template cast<bool>());
    }
    else if(py::isinstance<py::int_>(x) or py::hasattr(x, "__index__"))
    {
        f(x.template cast<int>());
    }
    else if(py::isinstance<py::float_>(x))
    {
        f(x.template cast<float>());
    }
    else if(py::isinstance<py::str>(x))
    {
        f(x.template cast<std::string>());
    }
    else if(py::isinstance<migraphx::shape::dynamic_dimension>(x))
    {
        f(migraphx::to_value(x.template cast<migraphx::shape::dynamic_dimension>()));
    }
    else
    {
        MIGRAPHX_THROW("VISIT_PY: Unsupported data type!");
    }
}

migraphx::value to_value(py::list lst)
{
    migraphx::value v = migraphx::value::array{};
    for(auto val : lst)
    {
        visit_py(val, [&](auto py_val) { v.push_back(py_val); });
    }

    return v;
}

migraphx::value to_value(py::kwargs kwargs)
{
    migraphx::value v = migraphx::value::object{};

    for(auto arg : kwargs)
    {
        auto&& key = py::str(arg.first);
        auto&& val = arg.second;
        visit_py(val, [&](auto py_val) { v[key] = py_val; });
    }
    return v;
}
} // namespace migraphx

namespace pybind11 {
namespace detail {

template <>
struct npy_format_descriptor<half>
{
    static std::string format()
    {
        // following: https://docs.python.org/3/library/struct.html#format-characters
        return "e";
    }
    static constexpr auto name() { return _("half"); }
};

template <>
struct npy_format_descriptor<migraphx::fp8::fp8e4m3fnuz>
{
    static std::string format()
    {
        // following: https://docs.python.org/3/library/struct.html#format-characters
        // TODO: need to figure out correct encoding
        return "z";
    }
    static constexpr auto name() { return _("fp8e4m3fnuz"); }
};

} // namespace detail
} // namespace pybind11

template <class F>
void visit_type(const migraphx::shape& s, F f)
{
    s.visit_type(f);
}

template <class T, class F>
void visit(const migraphx::raw_data<T>& x, F f)
{
    x.visit(f);
}

template <class F>
void visit_types(F f)
{
    migraphx::shape::visit_types(f);
}

template <class T>
py::buffer_info to_buffer_info(T& x)
{
    migraphx::shape s = x.get_shape();
    assert(s.type() != migraphx::shape::tuple_type);
    if(s.dynamic())
        MIGRAPHX_THROW("MIGRAPHX PYTHON: dynamic shape argument passed to to_buffer_info");
    auto strides = s.strides();
    std::transform(
        strides.begin(), strides.end(), strides.begin(), [&](auto i) { return i * s.type_size(); });
    py::buffer_info b;
    visit_type(s, [&](auto as) {
        // migraphx use int8_t data to store bool type, we need to
        // explicitly specify the data type as bool for python
        if(s.type() == migraphx::shape::bool_type)
        {
            b = py::buffer_info(x.data(),
                                as.size(),
                                py::format_descriptor<bool>::format(),
                                s.ndim(),
                                s.lens(),
                                strides);
        }
        else
        {
            b = py::buffer_info(x.data(),
                                as.size(),
                                py::format_descriptor<decltype(as())>::format(),
                                s.ndim(),
                                s.lens(),
                                strides);
        }
    });
    return b;
}

migraphx::shape to_shape(const py::buffer_info& info)
{
    migraphx::shape::type_t t;
    std::size_t n = 0;
    visit_types([&](auto as) {
        if(info.format == py::format_descriptor<decltype(as())>::format() or
           (info.format == "l" and py::format_descriptor<decltype(as())>::format() == "q") or
           (info.format == "L" and py::format_descriptor<decltype(as())>::format() == "Q"))
        {
            t = as.type_enum();
            n = sizeof(as());
        }
        else if(info.format == "?" and py::format_descriptor<decltype(as())>::format() == "b")
        {
            t = migraphx::shape::bool_type;
            n = sizeof(bool);
        }
    });

    if(n == 0)
    {
        MIGRAPHX_THROW("MIGRAPHX PYTHON: Unsupported data type " + info.format);
    }

    auto strides = info.strides;
    std::transform(strides.begin(), strides.end(), strides.begin(), [&](auto i) -> std::size_t {
        return n > 0 ? i / n : 0;
    });

    // scalar support
    if(info.shape.empty())
    {
        return migraphx::shape{t};
    }
    else
    {
        return migraphx::shape{t, info.shape, strides};
    }
}

int test_fun(int i, int j) { return i * j; }

void throwPyError(PyObject* type, std::string const& message)
{
    PyErr_SetString(type, message.data());
    throw py::error_already_set();
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

// TODO, figure out why this is needed and get rid of it
mgxinfer1::IBuilder* createInferBuilderWrapper(mgxinfer1::ILogger& logger)
{
    return createInferBuilder(logger);
}

mgxinfer1::IRuntime* createInferRuntimeWrapper(mgxinfer1::ILogger& logger)
{
    return createInferRuntime(logger);
}

MIGRAPHX_PYBIND11_MODULE(migraphx, m)
{
    using namespace pybind11::literals;

    auto common_api = m.def_submodule("common_api");

    /*Types start*/
    py::enum_<mgxinfer1::DataType>(common_api, "DataType", "TODO docstring", py::module_local())
        .value("FLOAT", mgxinfer1::DataType::kFLOAT, "TODO docstring")
        .value("HALF", mgxinfer1::DataType::kHALF, "TODO docstring")
        .value("BF16", mgxinfer1::DataType::kBF16, "TODO dosctring")
        .value("INT8", mgxinfer1::DataType::kINT8, "TODO dosctring")
        .value("INT32", mgxinfer1::DataType::kINT32, "TODO dosctring")
        .value("INT64", mgxinfer1::DataType::kINT64, "TODO dosctring")
        .value("BOOL", mgxinfer1::DataType::kBOOL, "TODO dosctring")
        .value("UINT8", mgxinfer1::DataType::kUINT8, "TODO dosctring")
        .value("FP8", mgxinfer1::DataType::kFP8, "TODO dosctring")
        .value("INT4", mgxinfer1::DataType::kINT4, "TODO dosctring");

    common_api.attr("float32")  = mgxinfer1::DataType::kFLOAT;
    common_api.attr("float16")  = mgxinfer1::DataType::kHALF;
    common_api.attr("bfloat16") = mgxinfer1::DataType::kBF16;
    common_api.attr("int8")     = mgxinfer1::DataType::kINT8;
    common_api.attr("int32")    = mgxinfer1::DataType::kINT32;
    common_api.attr("int64")    = mgxinfer1::DataType::kINT64;
    common_api.attr("bool")     = mgxinfer1::DataType::kBOOL;
    common_api.attr("uint8")    = mgxinfer1::DataType::kUINT8;
    common_api.attr("fp8")      = mgxinfer1::DataType::kFP8;
    common_api.attr("int4")     = mgxinfer1::DataType::kINT4;

    py::enum_<mgxinfer1::TensorIOMode>(
        common_api, "TensorIOMode", "TODO docstring", py::module_local())
        .value("NONE", mgxinfer1::TensorIOMode::kNONE, "TODO docstring")
        .value("INPUT", mgxinfer1::TensorIOMode::kINPUT, "TODO docstring")
        .value("OUTPUT", mgxinfer1::TensorIOMode::kOUTPUT, "TODO docstring");

    py::enum_<mgxinfer1::ExecutionContextAllocationStrategy>(common_api,
                                                             "ExecutionContextAllocationStrategy",
                                                             py::arithmetic{},
                                                             "TODO docstring",
                                                             py::module_local())
        .value("STATIC", mgxinfer1::ExecutionContextAllocationStrategy::kSTATIC, "TODO docstring")
        .value("ON_PROFILE_CHANGE",
               mgxinfer1::ExecutionContextAllocationStrategy::kON_PROFILE_CHANGE,
               "TODO docstring")
        .value("USER_MANAGED",
               mgxinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED,
               "TODO docstring");
    /*Types end*/

    /*Dims start*/
    constexpr auto dims_from_vec = [](std::vector<int64_t> const& in) {
        int32_t const maxDims{static_cast<int32_t>(mgxinfer1::Dims::MAX_DIMS)};
        PY_ASSERT_VALUE_ERROR(in.size() <= maxDims,
                              "Input length " + std::to_string(in.size()) +
                                  ". Max expected length is " + std::to_string(maxDims));

        mgxinfer1::Dims* dims = new mgxinfer1::Dims{};
        dims->nbDims          = in.size();
        for(int32_t i = 0; i < in.size(); ++i)
            dims->d[i] = in[i];
        return dims;
    };

    constexpr auto dims_getitem = [](const mgxinfer1::Dims& dims,
                                     const int32_t py_idx) -> const int64_t& {
        const int32_t idx{(py_idx < 0) ? static_cast<int32_t>(dims.nbDims) + py_idx : py_idx};
        PY_ASSERT_INDEX_ERROR(idx >= 0 && idx < dims.nbDims);
        return dims.d[idx];
    };

    constexpr auto dims_getitem_slice = [](const mgxinfer1::Dims& dims, py::slice slice) {
        size_t start, stop, step, slice_len;
        PY_ASSERT_VALUE_ERROR(slice.compute(dims.nbDims, &start, &stop, &step, &slice_len),
                              "Incorrect getter slice dims");
        PY_ASSERT_INDEX_ERROR(stop <= dims.nbDims);

        py::tuple ret{slice_len};
        for(int32_t i = start, idx = 0; i < stop; i += step, ++idx)
            ret[idx] = dims.d[i];
        return ret;
    };

    constexpr auto dims_to_str = [](const mgxinfer1::Dims& dims) {
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

    constexpr auto dims_setitem =
        [](mgxinfer1::Dims& dims, const int32_t py_idx, const int64_t item) {
            const int32_t idx{(py_idx < 0) ? static_cast<int32_t>(dims.nbDims) + py_idx : py_idx};
            PY_ASSERT_INDEX_ERROR(idx >= 0 && idx < dims.nbDims);
            dims.d[idx] = item;
        };

    constexpr auto dims_setitem_slice =
        [](mgxinfer1::Dims& dims, py::slice slice, const mgxinfer1::Dims& other) {
            size_t start, stop, step, slice_len;
            PY_ASSERT_VALUE_ERROR(slice.compute(dims.nbDims, &start, &stop, &step, &slice_len),
                                  "Incorrect setter slice dims");
            // Disallow out-of-bounds things.
            PY_ASSERT_INDEX_ERROR(stop < dims.nbDims);

            for(int32_t i = start, index = 0; i < stop; i += step, ++index)
                dims.d[i] = other.d[index];
        };

    py::class_<mgxinfer1::Dims>(common_api, "Dims", "TODO docstring", py::module_local())
        .def(py::init<>())
        // Allows for construction from python lists and tuples.
        .def(py::init(dims_from_vec), "shape"_a)
        // static_cast is required here, or MAX_DIMS does not get pulled in until LOAD time.
        .def_property_readonly_static(
            "MAX_DIMS",
            [](py::object) { return static_cast<int32_t const>(mgxinfer1::Dims::MAX_DIMS); },
            "TODO docstring")
        .def("__len__", [](const mgxinfer1::Dims& dims) { return dims.nbDims; })
        .def("__getitem__", dims_getitem)
        .def("__getitem__", dims_getitem_slice)
        .def("__setitem__", dims_setitem)
        .def("__setitem__", dims_setitem_slice)
        .def("__str__", dims_to_str)
        .def("__repr__", dims_to_str);

    // Make it possible to use tuples/lists in Python in place of Dims.
    py::implicitly_convertible<std::vector<int64_t>, mgxinfer1::Dims>();

    // TODO make this work for any python iterable
    common_api.def("volume", [](const mgxinfer1::Dims& dims) {
        size_t ret = 1;
        for(auto i = 0; i < dims.nbDims; ++i)
            ret *= dims.d[i];

        return ret;
    });

    /*Dims end*/

    /*Logger start*/
    // Trampoline for ILogger
    // https://pybind11.readthedocs.io/en/stable/advanced/classes.html
    class PyLogger : public mgxinfer1::ILogger
    {
        public:
        virtual void log(Severity severity, const char* msg) noexcept override
        {
            // TODO
            PYBIND11_OVERRIDE_PURE_NAME(void, mgxinfer1::ILogger, "log", log, severity, msg);
        }
    };

    auto ilogger =
        py::class_<mgxinfer1::ILogger, PyLogger>(
            common_api, "ILogger", "TODO dosctring", py::module_local())
            .def(py::init<>())
            .def("log", &mgxinfer1::ILogger::log, "severity"_a, "msg"_a, "TODO dosctring");

    py::enum_<mgxinfer1::ILogger::Severity>(
        ilogger, "Severity", py::arithmetic(), "TODO dosctring", py::module_local())
        .value("INTERNAL_ERROR", mgxinfer1::ILogger::Severity::kINTERNAL_ERROR, "TODO dosctring")
        .value("ERROR", mgxinfer1::ILogger::Severity::kERROR, "TODO dosctring")
        .value("WARNING", mgxinfer1::ILogger::Severity::kWARNING, "TODO dosctring")
        .value("INFO", mgxinfer1::ILogger::Severity::kINFO, "TODO dosctring")
        .value("VERBOSE", mgxinfer1::ILogger::Severity::kVERBOSE, "TODO dosctring")
        .export_values();

    py::class_<mgxinfer1::PythonLogger, mgxinfer1::ILogger>(
        common_api, "Logger", "TODO dosctring", py::module_local())
        .def(py::init<mgxinfer1::ILogger::Severity>(),
             "min_severity"_a = mgxinfer1::ILogger::Severity::kWARNING)
        .def("log", &mgxinfer1::PythonLogger::log, "severity"_a, "msg"_a, "TODO dosctring");
    /*Logger end*/

    /*INetworkDefinition start*/
    py::class_<mgxinfer1::INetworkDefinition>(
        common_api, "INetworkDefinition", "TODO docstring", py::module_local());

    /*IBuilder start*/
    py::class_<mgxinfer1::IHostMemory>(
        common_api, "IHostMemory", py::buffer_protocol(), "TODO docstring", py::module_local())
        .def_property_readonly("dtype",
                               [](mgxinfer1::IHostMemory const& mem) { return mem.type(); })
        .def_property_readonly("nbytes",
                               [](mgxinfer1::IHostMemory const& mem) { return mem.size(); })
        .def_buffer([](mgxinfer1::IHostMemory& mem) {
            py::buffer_info mem_info;
            mem_info.ptr      = mem.data();
            mem_info.itemsize = mgxinfer1::sizeofDataType(mem.type());
            // TODO this should be based on mem.type
            mem_info.format  = py::format_descriptor<char>::format();
            mem_info.ndim    = 1;
            mem_info.shape   = {static_cast<py::ssize_t>(mem.size())};
            mem_info.strides = {mem_info.itemsize};
            return mem_info;
        });

    py::enum_<mgxinfer1::MemoryPoolType>(
        common_api, "MemoryPoolType", "TODO docstring", py::module_local())
        .value("WORKSPACE", mgxinfer1::MemoryPoolType::kWORKSPACE, "TODO docstring")
        .value("DLA_MANAGED_SRAM", mgxinfer1::MemoryPoolType::kDLA_MANAGED_SRAM, "TODO docstring")
        .value("DLA_LOCAL_DRAM", mgxinfer1::MemoryPoolType::kDLA_LOCAL_DRAM, "TODO docstring")
        .value("DLA_GLOBAL_DRAM", mgxinfer1::MemoryPoolType::kDLA_GLOBAL_DRAM, "TODO docstring")
        .value("TACTIC_DRAM", mgxinfer1::MemoryPoolType::kTACTIC_DRAM, "TODO docstring")
        .value("TACTIC_SHARED_MEMORY",
               mgxinfer1::MemoryPoolType::kTACTIC_SHARED_MEMORY,
               "TODO docstring");

    py::class_<mgxinfer1::IBuilderConfig>(
        common_api, "IBuilderConfig", "TODO docstring", py::module_local())
        .def("set_memory_pool_limit",
             &mgxinfer1::IBuilderConfig::setMemoryPoolLimit,
             "pool"_a,
             "pool_size"_a,
             "TODO docstring");

    py::class_<mgxinfer1::IBuilder>(common_api, "Builder", "TODO docstring", py::module_local())
        .def(py::init(&createInferBuilderWrapper),
             "logger"_a,
             "TODO docstring",
             py::keep_alive<1, 2>{})
        .def("create_network",
             &mgxinfer1::IBuilder::createNetworkV2,
             "flags"_a = 0U,
             "TODO docstring",
             py::keep_alive<0, 1>{})
        .def("create_builder_config",
             &mgxinfer1::IBuilder::createBuilderConfig,
             "TODO docstring",
             py::keep_alive<0, 1>{})
        .def("build_serialized_network",
             &mgxinfer1::IBuilder::buildSerializedNetwork,
             "network"_a,
             "config"_a,
             "TODO docstring",
             py::call_guard<py::gil_scoped_release>{});
    /*IBuilder end*/

    /*ICudaEngine start*/
    py::class_<mgxinfer1::ICudaEngine>(
        common_api, "ICudaEngine", "TODO docstring", py::module_local())
        .def_property_readonly("num_io_tensors", &mgxinfer1::ICudaEngine::getNbIOTensors)
        .def("get_tensor_name",
             &mgxinfer1::ICudaEngine::getIOTensorName,
             "index"_a,
             "TODO docstring")
        .def(
            "get_tensor_shape", &mgxinfer1::ICudaEngine::getTensorShape, "name"_a, "TODO docstring")
        .def("get_tensor_dtype",
             &mgxinfer1::ICudaEngine::getTensorDataType,
             "name"_a,
             "TODO docstring")
        .def(
            "get_tensor_mode", &mgxinfer1::ICudaEngine::getTensorIOMode, "name"_a, "TODO docstring")
        .def("create_execution_context",
             &mgxinfer1::ICudaEngine::createExecutionContext,
             "TODO docstring",
             py::arg("strategy") = mgxinfer1::ExecutionContextAllocationStrategy::kSTATIC,
             py::keep_alive<0, 1>{},
             py::call_guard<py::gil_scoped_release>{});
    /*ICudaEngine end*/

    /*Runtime start*/
    const auto deserialize_engine_from_py_buffer = [](mgxinfer1::IRuntime& runtime,
                                                      py::buffer& serializedEngine) {
        py::buffer_info info = serializedEngine.request();
        return runtime.deserializeCudaEngine(info.ptr, info.size * info.itemsize);
    };

    py::class_<mgxinfer1::IRuntime>(common_api, "Runtime", "TODO docstring", py::module_local())
        .def(py::init(&createInferRuntimeWrapper),
             "logger"_a,
             "TODO docstring",
             py::keep_alive<1, 2>{})
        .def("deserialize_cuda_engine",
             deserialize_engine_from_py_buffer,
             "serialized_engine"_a,
             "TODO docstring",
             py::call_guard<py::gil_scoped_release>{},
             py::keep_alive<0, 1>{});
    /*Runtime end*/

    /*ExecutionContext start*/
    py::class_<mgxinfer1::IExecutionContext>(
        common_api, "IExecutionContext", "TODO docstring", py::module_local())
        .def(
            "execute_async_v3",
            [](mgxinfer1::IExecutionContext& context, size_t streamHandle) {
                return context.enqueueV3(reinterpret_cast<hipStream_t>(streamHandle));
            },
            "stream_handle"_a,
            "TODO docstring",
            py::call_guard<py::gil_scoped_release>{})
        .def(
            "set_tensor_address",
            [](mgxinfer1::IExecutionContext& context, char const* tensor_name, size_t memory) {
                return context.setTensorAddress(tensor_name, reinterpret_cast<void*>(memory));
            },
            "name"_a,
            "memory"_a,
            "TODO docstring");
    /*ExecutionContext end*/

    /*OnnxParser start*/
    const auto parse_from_py_buffer =
        [](mgxonnxparser::IParser& parser, const py::buffer& model, const char* path = nullptr) {
            py::buffer_info info = model.request();
            return parser.parse(info.ptr, info.size * info.itemsize, path);
        };

    py::class_<mgxonnxparser::IParser>(
        common_api, "OnnxParser", "TODO docstring", py::module_local())
        .def(py::init(&mgxonnxparser::createParser),
             "network"_a,
             "logger"_a,
             "TODO docstring",
             py::keep_alive<1, 3>{},
             py::keep_alive<2, 1>{})
        .def("parse",
             parse_from_py_buffer,
             "model"_a,
             "path"_a = nullptr,
             "TODO docstring",
             py::call_guard<py::gil_scoped_release>{})
        .def_property_readonly("num_errors", &mgxonnxparser::IParser::getNbErrors);
    /*OnnxParser end*/

    ////////////////////////////////
    ////////////////////////////////
    ////////////////////////////////
    ////////////////////////////////
    py::class_<migraphx::shape> shape_cls(m, "shape");
    shape_cls
        .def(py::init([](py::kwargs kwargs) {
            auto v = migraphx::to_value(kwargs);
            auto t = migraphx::shape::parse_type(v.get("type", "float"));
            if(v.contains("dyn_dims"))
            {
                auto dyn_dims =
                    migraphx::from_value<std::vector<migraphx::shape::dynamic_dimension>>(
                        v.at("dyn_dims"));
                return migraphx::shape(t, dyn_dims);
            }
            auto lens = v.get<std::size_t>("lens", {1});
            if(v.contains("strides"))
                return migraphx::shape(t, lens, v.at("strides").to_vector<std::size_t>());
            else
                return migraphx::shape(t, lens);
        }))
        .def("type", &migraphx::shape::type)
        .def("lens", &migraphx::shape::lens)
        .def("strides", &migraphx::shape::strides)
        .def("ndim", &migraphx::shape::ndim)
        .def("elements", &migraphx::shape::elements)
        .def("bytes", &migraphx::shape::bytes)
        .def("type_string", &migraphx::shape::type_string)
        .def("type_size", &migraphx::shape::type_size)
        .def("dyn_dims", &migraphx::shape::dyn_dims)
        .def("packed", &migraphx::shape::packed)
        .def("transposed", &migraphx::shape::transposed)
        .def("broadcasted", &migraphx::shape::broadcasted)
        .def("standard", &migraphx::shape::standard)
        .def("scalar", &migraphx::shape::scalar)
        .def("dynamic", &migraphx::shape::dynamic)
        .def("__eq__", std::equal_to<migraphx::shape>{})
        .def("__ne__", std::not_equal_to<migraphx::shape>{})
        .def("__repr__", [](const migraphx::shape& s) { return migraphx::to_string(s); });

    py::enum_<migraphx::shape::type_t>(shape_cls, "type_t")
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_PYTHON_GENERATE_SHAPE_ENUM);

    py::class_<migraphx::shape::dynamic_dimension>(shape_cls, "dynamic_dimension")
        .def(py::init<>())
        .def(py::init<std::size_t, std::size_t>())
        .def(py::init<std::size_t, std::size_t, std::set<std::size_t>>())
        .def_readwrite("min", &migraphx::shape::dynamic_dimension::min)
        .def_readwrite("max", &migraphx::shape::dynamic_dimension::max)
        .def_readwrite("optimals", &migraphx::shape::dynamic_dimension::optimals)
        .def("is_fixed", &migraphx::shape::dynamic_dimension::is_fixed);

    py::class_<migraphx::argument>(m, "argument", py::buffer_protocol())
        .def_buffer([](migraphx::argument& x) -> py::buffer_info { return to_buffer_info(x); })
        .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            return migraphx::argument(to_shape(info), info.ptr);
        }))
        .def("get_shape", &migraphx::argument::get_shape)
        .def("data_ptr",
             [](migraphx::argument& x) { return reinterpret_cast<std::uintptr_t>(x.data()); })
        .def("tolist",
             [](migraphx::argument& x) {
                 py::list l{x.get_shape().elements()};
                 visit(x, [&](auto data) { l = py::cast(data.to_vector()); });
                 return l;
             })
        .def("__eq__", std::equal_to<migraphx::argument>{})
        .def("__ne__", std::not_equal_to<migraphx::argument>{})
        .def("__repr__", [](const migraphx::argument& x) { return migraphx::to_string(x); });

    py::class_<migraphx::target>(m, "target");

    py::class_<migraphx::instruction_ref>(m, "instruction_ref")
        .def("shape", [](migraphx::instruction_ref i) { return i->get_shape(); })
        .def("op", [](migraphx::instruction_ref i) { return i->get_operator(); });

    py::class_<migraphx::module, std::unique_ptr<migraphx::module, py::nodelete>>(m, "module")
        .def("print", [](const migraphx::module& mm) { std::cout << mm << std::endl; })
        .def(
            "add_instruction",
            [](migraphx::module& mm,
               const migraphx::operation& op,
               std::vector<migraphx::instruction_ref>& args,
               std::vector<migraphx::module*>& mod_args) {
                return mm.add_instruction(op, args, mod_args);
            },
            py::arg("op"),
            py::arg("args"),
            py::arg("mod_args") = std::vector<migraphx::module*>{})
        .def(
            "add_literal",
            [](migraphx::module& mm, py::buffer data) {
                py::buffer_info info = data.request();
                auto literal_shape   = to_shape(info);
                return mm.add_literal(literal_shape, reinterpret_cast<char*>(info.ptr));
            },
            py::arg("data"))
        .def(
            "add_parameter",
            [](migraphx::module& mm, const std::string& name, const migraphx::shape shape) {
                return mm.add_parameter(name, shape);
            },
            py::arg("name"),
            py::arg("shape"))
        .def(
            "add_return",
            [](migraphx::module& mm, std::vector<migraphx::instruction_ref>& args) {
                return mm.add_return(args);
            },
            py::arg("args"))
        .def("__repr__", [](const migraphx::module& mm) { return migraphx::to_string(mm); });

    py::class_<migraphx::program>(m, "program")
        .def(py::init([]() { return migraphx::program(); }))
        .def("get_parameter_names", &migraphx::program::get_parameter_names)
        .def("get_parameter_shapes", &migraphx::program::get_parameter_shapes)
        .def("get_output_shapes", &migraphx::program::get_output_shapes)
        .def("is_compiled", &migraphx::program::is_compiled)
        .def(
            "compile",
            [](migraphx::program& p,
               const migraphx::target& t,
               bool offload_copy,
               bool fast_math,
               bool exhaustive_tune) {
                migraphx::compile_options options;
                options.offload_copy    = offload_copy;
                options.fast_math       = fast_math;
                options.exhaustive_tune = exhaustive_tune;
                p.compile(t, options);
            },
            py::arg("t"),
            py::arg("offload_copy")    = true,
            py::arg("fast_math")       = true,
            py::arg("exhaustive_tune") = false)
        .def("get_main_module", [](const migraphx::program& p) { return p.get_main_module(); })
        .def(
            "create_module",
            [](migraphx::program& p, const std::string& name) { return p.create_module(name); },
            py::arg("name"))
        .def("run",
             [](migraphx::program& p, py::dict params) {
                 migraphx::parameter_map pm;
                 for(auto x : params)
                 {
                     std::string key      = x.first.cast<std::string>();
                     py::buffer b         = x.second.cast<py::buffer>();
                     py::buffer_info info = b.request();
                     pm[key]              = migraphx::argument(to_shape(info), info.ptr);
                 }
                 return p.eval(pm);
             })
        .def("run_async",
             [](migraphx::program& p,
                py::dict params,
                std::uintptr_t stream,
                std::string stream_name) {
                 migraphx::parameter_map pm;
                 for(auto x : params)
                 {
                     std::string key      = x.first.cast<std::string>();
                     py::buffer b         = x.second.cast<py::buffer>();
                     py::buffer_info info = b.request();
                     pm[key]              = migraphx::argument(to_shape(info), info.ptr);
                 }
                 migraphx::execution_environment exec_env{
                     migraphx::any_ptr(reinterpret_cast<void*>(stream), stream_name), true};
                 return p.eval(pm, exec_env);
             })
        .def("sort", &migraphx::program::sort)
        .def("print", [](const migraphx::program& p) { std::cout << p << std::endl; })
        .def("__eq__", std::equal_to<migraphx::program>{})
        .def("__ne__", std::not_equal_to<migraphx::program>{})
        .def("__repr__", [](const migraphx::program& p) { return migraphx::to_string(p); });

    py::class_<migraphx::operation> op(m, "op");
    op.def(py::init([](const std::string& name, py::kwargs kwargs) {
          migraphx::value v = migraphx::value::object{};
          if(kwargs)
          {
              v = migraphx::to_value(kwargs);
          }
          return migraphx::make_op(name, v);
      }))
        .def("name", &migraphx::operation::name);

    py::enum_<migraphx::op::pooling_mode>(op, "pooling_mode")
        .value("average", migraphx::op::pooling_mode::average)
        .value("max", migraphx::op::pooling_mode::max)
        .value("lpnorm", migraphx::op::pooling_mode::lpnorm);

    py::enum_<migraphx::op::rnn_direction>(op, "rnn_direction")
        .value("forward", migraphx::op::rnn_direction::forward)
        .value("reverse", migraphx::op::rnn_direction::reverse)
        .value("bidirectional", migraphx::op::rnn_direction::bidirectional);

    m.def(
        "argument_from_pointer",
        [](const migraphx::shape shape, const int64_t address) {
            return migraphx::argument(shape, reinterpret_cast<void*>(address));
        },
        py::arg("shape"),
        py::arg("address"));

    m.def(
        "parse_tf",
        [](const std::string& filename,
           bool is_nhwc,
           unsigned int batch_size,
           std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims,
           std::vector<std::string> output_names) {
            return migraphx::parse_tf(
                filename, migraphx::tf_options{is_nhwc, batch_size, map_input_dims, output_names});
        },
        "Parse tf protobuf (default format is nhwc)",
        py::arg("filename"),
        py::arg("is_nhwc")        = true,
        py::arg("batch_size")     = 1,
        py::arg("map_input_dims") = std::unordered_map<std::string, std::vector<std::size_t>>(),
        py::arg("output_names")   = std::vector<std::string>());

    m.def(
        "parse_onnx",
        [](const std::string& filename,
           unsigned int default_dim_value,
           migraphx::shape::dynamic_dimension default_dyn_dim_value,
           std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims,
           std::unordered_map<std::string, std::vector<migraphx::shape::dynamic_dimension>>
               map_dyn_input_dims,
           bool skip_unknown_operators,
           bool print_program_on_error,
           int64_t max_loop_iterations,
           int64_t limit_max_iterations) {
            migraphx::onnx_options options;
            options.default_dim_value      = default_dim_value;
            options.default_dyn_dim_value  = default_dyn_dim_value;
            options.map_input_dims         = map_input_dims;
            options.map_dyn_input_dims     = map_dyn_input_dims;
            options.skip_unknown_operators = skip_unknown_operators;
            options.print_program_on_error = print_program_on_error;
            options.max_loop_iterations    = max_loop_iterations;
            options.limit_max_iterations   = limit_max_iterations;
            return migraphx::parse_onnx(filename, options);
        },
        "Parse onnx file",
        py::arg("filename"),
        py::arg("default_dim_value")     = 0,
        py::arg("default_dyn_dim_value") = migraphx::shape::dynamic_dimension{1, 1},
        py::arg("map_input_dims") = std::unordered_map<std::string, std::vector<std::size_t>>(),
        py::arg("map_dyn_input_dims") =
            std::unordered_map<std::string, std::vector<migraphx::shape::dynamic_dimension>>(),
        py::arg("skip_unknown_operators") = false,
        py::arg("print_program_on_error") = false,
        py::arg("max_loop_iterations")    = 10,
        py::arg("limit_max_iterations")   = std::numeric_limits<uint16_t>::max());

    m.def(
        "parse_onnx_buffer",
        [](const std::string& onnx_buffer,
           unsigned int default_dim_value,
           migraphx::shape::dynamic_dimension default_dyn_dim_value,
           std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims,
           std::unordered_map<std::string, std::vector<migraphx::shape::dynamic_dimension>>
               map_dyn_input_dims,
           bool skip_unknown_operators,
           bool print_program_on_error) {
            migraphx::onnx_options options;
            options.default_dim_value      = default_dim_value;
            options.default_dyn_dim_value  = default_dyn_dim_value;
            options.map_input_dims         = map_input_dims;
            options.map_dyn_input_dims     = map_dyn_input_dims;
            options.skip_unknown_operators = skip_unknown_operators;
            options.print_program_on_error = print_program_on_error;
            return migraphx::parse_onnx_buffer(onnx_buffer, options);
        },
        "Parse onnx file",
        py::arg("filename"),
        py::arg("default_dim_value")     = 0,
        py::arg("default_dyn_dim_value") = migraphx::shape::dynamic_dimension{1, 1},
        py::arg("map_input_dims") = std::unordered_map<std::string, std::vector<std::size_t>>(),
        py::arg("map_dyn_input_dims") =
            std::unordered_map<std::string, std::vector<migraphx::shape::dynamic_dimension>>(),
        py::arg("skip_unknown_operators") = false,
        py::arg("print_program_on_error") = false);

    m.def(
        "load",
        [](const std::string& name, const std::string& format) {
            migraphx::file_options options;
            options.format = format;
            return migraphx::load(name, options);
        },
        "Load MIGraphX program",
        py::arg("filename"),
        py::arg("format") = "msgpack");

    m.def(
        "save",
        [](const migraphx::program& p, const std::string& name, const std::string& format) {
            migraphx::file_options options;
            options.format = format;
            return migraphx::save(p, name, options);
        },
        "Save MIGraphX program",
        py::arg("p"),
        py::arg("filename"),
        py::arg("format") = "msgpack");

    m.def("get_target", &migraphx::make_target);
    m.def("create_argument", [](const migraphx::shape& s, const std::vector<double>& values) {
        if(values.size() != s.elements())
            MIGRAPHX_THROW("Values and shape elements do not match");
        migraphx::argument a{s};
        a.fill(values.begin(), values.end());
        return a;
    });
    m.def("generate_argument", &migraphx::generate_argument, py::arg("s"), py::arg("seed") = 0);
    m.def("fill_argument", &migraphx::fill_argument, py::arg("s"), py::arg("value"));
    m.def("quantize_fp16",
          &migraphx::quantize_fp16,
          py::arg("prog"),
          py::arg("ins_names") = std::vector<std::string>{"all"});
    m.def("quantize_int8",
          &migraphx::quantize_int8,
          py::arg("prog"),
          py::arg("t"),
          py::arg("calibration") = std::vector<migraphx::parameter_map>{},
          py::arg("ins_names")   = std::unordered_set<std::string>{"dot", "convolution"});
    m.def(
        "autocast_fp8",
        [](migraphx::program& prog) {
            migraphx::run_passes(*prog.get_main_module(), {migraphx::autocast_fp8_pass{}});
        },
        "Auto-convert FP8 parameters and return values to Float for MIGraphX Program",
        py::arg("prog"));

#ifdef HAVE_GPU
    m.def("allocate_gpu", &migraphx::gpu::allocate_gpu, py::arg("s"), py::arg("host") = false);
    m.def("to_gpu", &migraphx::gpu::to_gpu, py::arg("arg"), py::arg("host") = false);
    m.def("from_gpu", &migraphx::gpu::from_gpu);
    m.def("gpu_sync", [] { migraphx::gpu::gpu_sync(); });
#endif

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
