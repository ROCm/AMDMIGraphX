#include <migraphx/config.hpp>
#include <migraphx/program.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/file_buffer.hpp>
#include <pybind11/embed.h>
#include <pybind11/eval.h>

namespace py = pybind11;

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

extern "C" program migraphx_load_py(const std::string& filename);

const std::string& python_path()
{
    static const auto path = dynamic_loader::path(&migraphx_load_py).parent_path().string();
    return path;
}

static py::dict run_file(const std::string& file)
{
    py::object scope = py::module_::import("__main__").attr("__dict__");
    std::string buffer;
    buffer.append("import sys\n");
    buffer.append("sys.path.insert(0, '" + python_path() + "')\n");
    buffer.append("import migraphx\n");
    buffer.append(read_string(file));
    py::exec(buffer, scope);
    return scope.cast<py::dict>();
}


extern "C" program migraphx_load_py(const std::string& filename)
{
    py::scoped_interpreter guard{};
    py::dict vars = run_file(filename);
    auto it = std::find_if(vars.begin(), vars.end(), [](const auto& p) {
        return py::isinstance<migraphx::program>(p.second);
    });
    if(it == vars.end())
        MIGRAPHX_THROW("No program variable found");
    return it->second.cast<migraphx::program>();

}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

