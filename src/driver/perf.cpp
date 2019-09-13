#include "perf.hpp"

#include <migraphx/cpu/target.hpp>
#include <migraphx/generate.hpp>
#ifdef HAVE_GPU
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#endif

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

program::parameter_map fill_param_map(program::parameter_map& m, const program& p, bool gpu)
{
    for(auto&& x : p.get_parameter_shapes())
    {
        argument& arg = m[x.first];
        if(arg.empty())
            arg = generate_argument(x.second);
#ifdef HAVE_GPU
        if(gpu)
            arg = gpu::to_gpu(arg);
#else
        (void)gpu;
#endif
    }
    return m;
}

program::parameter_map create_param_map(const program& p, bool gpu)
{
    program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
#ifdef HAVE_GPU
        if(gpu)
            m[x.first] = gpu::to_gpu(generate_argument(x.second));
        else
#else
        (void)gpu;
#endif
            m[x.first] = generate_argument(x.second);
    }
    return m;
}


target get_target(bool gpu)
{
        if(gpu)
    {
#ifdef HAVE_GPU
        return gpu::target{};
#else
        MIGRAPHX_THROW("Gpu not supported.");
#endif
    }
    else
    {
        return cpu::target{};
    }
}

void compile_program(program& p, bool gpu)
{
    if(gpu)
    {
#ifdef HAVE_GPU
        p.compile(gpu::target{});
#else
        MIGRAPHX_THROW("Gpu not supported.");
#endif
    }
    else
    {
        p.compile(cpu::target{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
