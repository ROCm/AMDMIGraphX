
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <migraphx/program.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/type_name.hpp>

#ifdef HAVE_GPU
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#endif

namespace py = pybind11;

template <class F>
struct throw_half
{
    F f;

    template <class A>
    void operator()(A a) const
    {
        f(a);
    }

    void operator()(migraphx::shape::as<migraphx::half>) const
    {
        throw std::runtime_error("Half not supported in python yet.");
    }

    void operator()(migraphx::tensor_view<migraphx::half>) const
    {
        throw std::runtime_error("Half not supported in python yet.");
    }
};

template <class F>
struct skip_half
{
    F f;

    template <class A>
    void operator()(A a) const
    {
        f(a);
    }

    void operator()(migraphx::shape::as<migraphx::half>) const {}

    void operator()(migraphx::tensor_view<migraphx::half>) const {}
};

template <class F>
void visit_type(const migraphx::shape& s, F f)
{
    s.visit_type(throw_half<F>{f});
}

template <class T, class F>
void visit(const migraphx::raw_data<T>& x, F f)
{
    x.visit(throw_half<F>{f});
}

template <class F>
void visit_types(F f)
{
    migraphx::shape::visit_types(skip_half<F>{f});
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
        b = py::buffer_info(x.data(),
                            as.size(),
                            py::format_descriptor<decltype(as())>::format(),
                            s.lens().size(),
                            s.lens(),
                            strides);
    });
    return b;
}

migraphx::shape to_shape(const py::buffer_info& info)
{
    migraphx::shape::type_t t;
    std::size_t n = 0;
    visit_types([&](auto as) {
        if(info.format == py::format_descriptor<decltype(as())>::format())
        {
            t = as.type_enum();
            n = sizeof(as());
        }
    });

    if(n == 0)
    {
        MIGRAPHX_THROW("MIGRAPHX PYTHON: Unsupported data type" + info.format);
    }

    auto strides = info.strides;
    std::transform(strides.begin(), strides.end(), strides.begin(), [&](auto i) -> std::size_t {
        return n > 0 ? i / n : 0;
    });
    return migraphx::shape{t, info.shape, strides};
}

PYBIND11_MODULE(migraphx, m)
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

    py::class_<migraphx::program>(m, "program")
        .def("clone", [](migraphx::program& p) { return *(new migraphx::program(p)); })
        .def("get_parameter_shapes", &migraphx::program::get_parameter_shapes)
        .def("get_output_shapes", &migraphx::program::get_output_shapes)
        .def("compile",
             [](migraphx::program& p, const migraphx::target& t, bool offload_copy) {
                 migraphx::compile_options options;
                 options.offload_copy = offload_copy;
                 p.compile(t, options);
             },
             py::arg("t"),
             py::arg("offload_copy") = true)
        .def("run", &migraphx::program::eval)
        .def("__eq__", std::equal_to<migraphx::program>{})
        .def("__ne__", std::not_equal_to<migraphx::program>{})
        .def("__repr__", [](const migraphx::program& p) { return migraphx::to_string(p); });

    m.def("parse_tf",
          [](const std::string& filename, bool is_nhwc, unsigned int batch_size) {
              return migraphx::parse_tf(filename, migraphx::tf_options{is_nhwc, batch_size});
          },
          "Parse tf protobuf (default format is nhwc)",
          py::arg("filename"),
          py::arg("is_nhwc")    = true,
          py::arg("batch_size") = 1);

    m.def("parse_onnx",
          [](const std::string& filename,
             std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims,
             std::size_t value) {
              migraphx::onnx_options options;
              options.map_input_dims = map_input_dims;
              options.default_dim_value = value;
              return migraphx::parse_onnx(filename, options);
          },
          "Parse onnx file",
          py::arg("filename"),
          py::arg("map_input_dims") = std::map<std::string, std::vector<std::size_t>>(),
          py::arg("value")     = 1);

    m.def("get_target", [](const std::string& name) -> migraphx::target {
        if(name == "cpu")
            return migraphx::cpu::target{};
#ifdef HAVE_GPU
        if(name == "gpu")
            return migraphx::gpu::target{};
#endif
        throw std::runtime_error("Target not found: " + name);
    });

    m.def("generate_argument", &migraphx::generate_argument, py::arg("s"), py::arg("seed") = 0);
    m.def("quantize_fp16",
          &migraphx::quantize_fp16,
          py::arg("prog"),
          py::arg("ins_names") = std::vector<std::string>{"all"});
    m.def("quantize_int8",
          &migraphx::quantize_int8,
          py::arg("prog"),
          py::arg("t"),
          py::arg("calibration") = std::vector<migraphx::program::parameter_map>{},
          py::arg("ins_names")   = std::vector<std::string>{"dot", "convolution"});

#ifdef HAVE_GPU
    m.def("allocate_gpu", &migraphx::gpu::allocate_gpu, py::arg("s"), py::arg("host") = false);
    m.def("to_gpu", &migraphx::gpu::to_gpu, py::arg("arg"), py::arg("host") = false);
    m.def("from_gpu", &migraphx::gpu::from_gpu);
    m.def("gpu_sync", &migraphx::gpu::gpu_sync);
#endif

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
