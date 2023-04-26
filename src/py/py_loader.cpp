#include <migraphx/py.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/process.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static std::vector<fs::path> find_available_python_versions()
{
    std::vector<fs::path> result;
    auto path = dynamic_loader::path(&load_py).parent_path();
    for(const auto& entry : fs::directory_iterator{path})
    {
        if(not entry.is_regular_file())
            continue;
        auto p = entry.path();
        if(not contains(p.stem().string(), "migraphx_py_"))
            continue;
        result.push_back(p);
    }
    std::sort(result.begin(), result.end(), std::greater<>{});
    return result;
}

static dynamic_loader load_py_lib()
{
    auto libs = find_available_python_versions();
    for(const auto& lib : libs)
    {
        auto result = dynamic_loader::try_load(lib);
        if(result.has_value())
            return *result;
    }
    MIGRAPHX_THROW("Cant find a viable version of python");
}

static dynamic_loader py_lib()
{
    static dynamic_loader lib = load_py_lib();
    return lib;
}

program load_py(const std::string& filename)
{
    static auto f = py_lib().get_function<program(const std::string&)>("migraphx_load_py");
    return f(filename);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
