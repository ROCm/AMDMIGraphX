#ifndef MIGRAPHX_GUARD_RTGLIB_TRACER_HPP
#define MIGRAPHX_GUARD_RTGLIB_TRACER_HPP

#include <fstream>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct tracer
{
    tracer() {}

    tracer(std::string dump_directory) : dump_dir(dump_directory), counter(0) {}

    bool enabled() const { return !dump_dir.empty(); }

    template <class... Ts>
    void operator()(const std::string& program_name, const Ts&... xs) 
    {
        if(this->enabled()) {
            fs::path dir_path = fs::current_path() / this->dump_dir;
            if(not fs::exists(dir_path)) {
                fs::create_directories(dir_path);
            } 
            fs::path ir_file_path = dir_path / (std::to_string(this->counter++)+"_"+program_name+".mxr");
            std::ofstream ofs(ir_file_path);
            swallow{ofs<<xs...};
            ofs<<std::endl;
            ofs.close();
        }
    }
    std::string dump_dir;
    private:
    uint counter;
};


} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
