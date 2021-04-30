
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <migraphx/program.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/type_name.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/json.hpp>
#include <migraphx/make_op.hpp>

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
    else if(py::isinstance<py::int_>(x))
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
    auto strides      = s.strides();
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
                                s.lens().size(),
                                s.lens(),
                                strides);
        }
        else
        {
            b = py::buffer_info(x.data(),
                                as.size(),
                                py::format_descriptor<decltype(as())>::format(),
                                s.lens().size(),
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

MIGRAPHX_PYBIND11_MODULE(migraphx, m)
{
    py::class_<migraphx::shape>(m, "shape")
        .def(py::init<>())
        .def("type", &migraphx::shape::type)
        .def("lens", &migraphx::shape::lens)
        .def("strides", &migraphx::shape::strides)
        .def("elements", &migraphx::shape::elements)
        .def("bytes", &migraphx::shape::bytes)
        .def("type_size", &migraphx::shape::type_size)
        .def("packed", &migraphx::shape::packed)
        .def("transposed", &migraphx::shape::transposed)
        .def("broadcasted", &migraphx::shape::broadcasted)
        .def("standard", &migraphx::shape::standard)
        .def("scalar", &migraphx::shape::scalar)
        .def("__eq__", std::equal_to<migraphx::shape>{})
        .def("__ne__", std::not_equal_to<migraphx::shape>{})
        .def("__repr__", [](const migraphx::shape& s) { return migraphx::to_string(s); });

    py::class_<migraphx::argument>(m, "argument", py::buffer_protocol())
        .def_buffer([](migraphx::argument& x) -> py::buffer_info { return to_buffer_info(x); })
        .def("__init__",
             [](migraphx::argument& x, py::buffer b) {
                 py::buffer_info info = b.request();
                 new(&x) migraphx::argument(to_shape(info), info.ptr);
             })
        .def("get_shape", &migraphx::argument::get_shape)
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

    py::class_<migraphx::module>(m, "module")
        .def("print", [](const migraphx::module& mm) { std::cout << mm << std::endl; })
        .def("__eq__", std::equal_to<migraphx::module>{})
        .def("__ne__", std::not_equal_to<migraphx::module>{})
        .def("__repr__", [](const migraphx::module& mm) { return migraphx::to_string(mm); });

    py::class_<migraphx::program>(m, "program")
        .def("get_parameter_names", &migraphx::program::get_parameter_names)
        .def("get_parameter_shapes", &migraphx::program::get_parameter_shapes)
        .def("get_output_shapes", &migraphx::program::get_output_shapes)
        .def(
            "compile",
            [](migraphx::program& p, const migraphx::target& t, bool offload_copy, bool fast_math) {
                migraphx::compile_options options;
                options.offload_copy = offload_copy;
                options.fast_math    = fast_math;
                p.compile(t, options);
            },
            py::arg("t"),
            py::arg("offload_copy") = true,
            py::arg("fast_math")    = true)
        .def("get_main_module",
             [](migraphx::program& p) {
                 auto* mm = p.get_main_module();
                 return *mm;
             })
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
        .def("sort", &migraphx::program::sort)
        .def("print", [](const migraphx::program& p) { std::cout << p << std::endl; })
        .def("__eq__", std::equal_to<migraphx::program>{})
        .def("__ne__", std::not_equal_to<migraphx::program>{})
        .def("__repr__", [](const migraphx::program& p) { return migraphx::to_string(p); });

    py::class_<migraphx::operation>(m, "op")
        .def(py::init([](const std::string& name, py::kwargs kwargs) {
            migraphx::value v = migraphx::value::object{};
            if(kwargs)
            {
                v = migraphx::to_value(kwargs);
            }
            return migraphx::make_op(name, v);
        }))

        .def("name", &migraphx::operation::name);

    m.def("parse_tf",
          [](const std::string& filename,
             bool is_nhwc,
             unsigned int batch_size,
             std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims,
             std::vector<std::string> output_names) {
              return migraphx::parse_tf(
                  filename,
                  migraphx::tf_options{is_nhwc, batch_size, map_input_dims, output_names});
          },
          "Parse tf protobuf (default format is nhwc)",
          py::arg("filename"),
          py::arg("is_nhwc")        = true,
          py::arg("batch_size")     = 1,
          py::arg("map_input_dims") = std::unordered_map<std::string, std::vector<std::size_t>>(),
          py::arg("output_names")   = std::vector<std::string>());

    m.def("parse_onnx",
          [](const std::string& filename,
             unsigned int default_dim_value,
             std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims,
             bool skip_unknown_operators,
             bool print_program_on_error) {
              migraphx::onnx_options options;
              options.default_dim_value      = default_dim_value;
              options.map_input_dims         = map_input_dims;
              options.skip_unknown_operators = skip_unknown_operators;
              options.print_program_on_error = print_program_on_error;
              return migraphx::parse_onnx(filename, options);
          },
          "Parse onnx file",
          py::arg("filename"),
          py::arg("default_dim_value") = 1,
          py::arg("map_input_dims") = std::unordered_map<std::string, std::vector<std::size_t>>(),
          py::arg("skip_unknown_operators") = false,
          py::arg("print_program_on_error") = false);

    m.def("parse_onnx_buffer",
          [](const std::string& onnx_buffer,
             unsigned int default_dim_value,
             std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims,
             bool skip_unknown_operators,
             bool print_program_on_error) {
              migraphx::onnx_options options;
              options.default_dim_value      = default_dim_value;
              options.map_input_dims         = map_input_dims;
              options.skip_unknown_operators = skip_unknown_operators;
              options.print_program_on_error = print_program_on_error;
              return migraphx::parse_onnx_buffer(onnx_buffer, options);
          },
          "Parse onnx file",
          py::arg("filename"),
          py::arg("default_dim_value") = 1,
          py::arg("map_input_dims") = std::unordered_map<std::string, std::vector<std::size_t>>(),
          py::arg("skip_unknown_operators") = false,
          py::arg("print_program_on_error") = false);

    m.def("load",
          [](const std::string& name, const std::string& format) {
              migraphx::file_options options;
              options.format = format;
              return migraphx::load(name, options);
          },
          "Load MIGraphX program",
          py::arg("filename"),
          py::arg("format") = "msgpack");

    m.def("save",
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
    m.def("generate_argument", &migraphx::generate_argument, py::arg("s"), py::arg("seed") = 0);
    m.def("quantize_fp16",
          &migraphx::quantize_fp16,
          py::arg("prog"),
          py::arg("ins_names") = std::vector<std::string>{"all"});
    m.def("quantize_int8",
          &migraphx::quantize_int8,
          py::arg("prog"),
          py::arg("t"),
          py::arg("calibration") = std::vector<migraphx::parameter_map>{},
          py::arg("ins_names")   = std::vector<std::string>{"dot", "convolution"});

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
