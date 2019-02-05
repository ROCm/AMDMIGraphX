
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
struct skip_half
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
void visit_type(const migraphx::shape& s, F f)
{
    s.visit_type(skip_half<F>{f});
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
        .def_buffer([](migraphx::argument& x) -> py::buffer_info { return to_buffer_info(x); });

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
