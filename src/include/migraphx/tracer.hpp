#ifndef MIGRAPHX_GUARD_RTGLIB_TRACER_HPP
#define MIGRAPHX_GUARD_RTGLIB_TRACER_HPP

#include <fstream>
#include <iostream>
#include <ostream>
#include <migraphx/requires.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct tracer
{
    tracer() {}

    tracer(std::ostream& s) : os(&s) {}

    tracer(const std::string& dump_directory)
        : dump_dir(dump_directory), counter(0), dir_path(fs::current_path() / dump_directory)
    {
        if(!dump_dir.empty() && fs::exists(dir_path))
        {
            fs::remove_all(dir_path);
        }
        fs::create_directories(dir_path);
    }
    // file_stream
    bool fs_enabled() const { return !dump_dir.empty() && !os_enabled(); }
    // output_stream
    bool os_enabled() const { return (os != nullptr) && !fs_enabled(); }
    bool enabled() const { return fs_enabled() or os_enabled(); };

    /*
    Dump any string to ostream, used for debug build or debugging purposes.
    */
    void operator()(const std::string& s = "") const { std::cout << s << std::endl; }

    /*
    Based on user's envrionment flags, either dump IR passes' output to a file or ostream i.e. cout
    or cerr, :param pass_file_name : file_name to be used when dumping IR pass to a file, this param
    is not used when IR is dumped to ostream.
    */
    template <class... Ts, MIGRAPHX_REQUIRES((sizeof...(Ts) > 0))>
    void operator()(const std::string& pass_file_name, const Ts&... xs)
    {
        if(fs_enabled())
        {
            fs::path ir_file_path =
                dir_path / (std::to_string(counter++) + "_" + pass_file_name + ".mxr");
            std::ofstream ofs(ir_file_path);
            swallow{ofs << xs...};
            ofs << std::endl;
            ofs.close();
        }
        else if(os_enabled())
        {
            swallow{*os << xs...};
            *os << std::endl;
        }
    }

    std::string dump_dir = "";

    private:
    uint counter      = 0;
    std::ostream* os  = nullptr;
    fs::path dir_path = "";
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
