
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/stringutils.hpp>
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

    void operator()(migraphx::shape::as<migraphx::half>) const
    {}
};

template <class F>
void visit_type(const migraphx::shape& s, F f)
{
    s.visit_type(throw_half<F>{f});
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
    py::buffer_info b;
    visit_type(s, [&](auto as) {
        b = py::buffer_info(x.data(),
                            as.size(),
                            py::format_descriptor<decltype(as())>::format(),
                            s.lens().size(),
                            s.lens(),
                            s.strides());
    });
    return b;
}

migraphx::shape to_shape(const py::buffer_info& info)
{
    migraphx::shape::type_t t;
    visit_types([&](auto as) {
        if (info.format == py::format_descriptor<decltype(as())>::format())
            t = as.type_enum();
    });
    return migraphx::shape{t, info.shape, info.strides};
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
        .def("__repr__", [](const migraphx::shape& s) { return migraphx::to_string(s); });

    py::class_<migraphx::argument>(m, "argument", py::buffer_protocol())
        .def_buffer([](migraphx::argument& x) -> py::buffer_info { return to_buffer_info(x); })
        .def("__init__", [](migraphx::argument& x, py::buffer b) {
            py::buffer_info info = b.request();
            new (&x) migraphx::argument(to_shape(info), info.ptr);
        });

    py::class_<migraphx::target>(m, "target");

    py::class_<migraphx::program>(m, "program")
        .def("get_parameter_shapes", &migraphx::program::get_parameter_shapes)
        .def("compile", [](migraphx::program& p, const migraphx::target& t) { p.compile(t); })
        .def("run", &migraphx::program::eval)
        .def("__repr__", [](const migraphx::program& p) { return migraphx::to_string(p); });

    m.def("parse_onnx", &migraphx::parse_onnx);

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

#ifdef HAVE_GPU
    m.def("allocate_gpu", &migraphx::gpu::allocate_gpu, py::arg("s"), py::arg("host") = false);
    m.def("to_gpu", &migraphx::gpu::to_gpu, py::arg("arg"), py::arg("host") = false);
    m.def("from_gpu", &migraphx::gpu::from_gpu);
    m.def("gpu_sync", &migraphx::gpu::gpu_sync);
    m.def("copy_to_gpu", &migraphx::gpu::copy_to_gpu);
#endif

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
