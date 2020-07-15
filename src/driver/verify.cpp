#include "verify.hpp"

#include <migraphx/cpu/target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify_args.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/compile_options.hpp>

#ifdef HAVE_GPU
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#endif

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

std::vector<argument> run_cpu(program p)
{
    p.compile(cpu::target{});
    program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = generate_argument(x.second, get_hash(x.first));
    }
    auto out = p.eval(m);
    std::cout << p << std::endl;
    return out;
}

std::vector<argument> run_gpu(program p, const compile_options& options)
{
#ifdef HAVE_GPU
    p.compile(gpu::target{}, options);

    program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        auto arg   = generate_argument(x.second, get_hash(x.first));
        m[x.first] = options.offload_copy ? arg : gpu::to_gpu(arg);
    }
    auto gpu_out = p.eval(m);
    std::vector<argument> output(gpu_out.size());
    std::cout << p << std::endl;
    std::transform(gpu_out.begin(), gpu_out.end(), output.begin(), [&](auto& argu) {
        return options.offload_copy ? argu : gpu::from_gpu(argu);
    });
    return output;

#else
    (void)p;
    MIGRAPHX_THROW("Gpu unsupported!");
#endif
}

void verify_program(const std::string& name,
                    const program& p,
                    compile_options options,
                    double tolerance)
{
    auto x = run_cpu(p);
    auto y = run_gpu(p, options);

    std::size_t output_num = x.size();
    for(std::size_t i = 0; i < output_num; ++i)
    {
        verify_args(name, x[i], y[i], tolerance);
    }
    // std::cout << "cpu: " << x << std::endl;
    // std::cout << "gpu: " << y << std::endl;
}

void verify_instructions(const program& prog, compile_options options, double tolerance)
{
    for(auto&& ins : prog)
    {
        if(ins.name().front() == '@')
            continue;
        if(ins.name() == "broadcast")
            continue;
        if(ins.name() == "transpose")
            continue;
        if(ins.name() == "reshape")
            continue;
        if(ins.name() == "undefined")
            continue;
        program p;
        std::vector<instruction_ref> inputs;
        for(auto&& arg : ins.inputs())
        {
            if(arg->name() == "@literal")
                inputs.push_back(p.add_literal(arg->get_literal()));
            else
                inputs.push_back(p.add_parameter(std::to_string(inputs.size()), arg->get_shape()));
        }
        p.add_instruction(ins.get_operator(), inputs);
        try
        {
            std::cout << "Verify: " << ins.name() << std::endl;
            std::cout << p << std::endl;
            verify_program(ins.name(), p, options, tolerance);
        }
        catch(...)
        {
            std::cout << "Instruction " << ins.name() << " threw an exception." << std::endl;
            throw;
        }
    }
}

void verify_reduced(program p, int n, compile_options options, double tolerance)
{
    auto last = std::prev(p.end(), n + 1);
    p.remove_instructions(last, p.end());
    std::cout << "Verify: " << std::endl;
    std::cout << p << std::endl;
    verify_program(std::to_string(n), p, options, tolerance);
}

void verify_reduced_program(const program& p, compile_options options, double tolerance)
{
    auto n = std::distance(p.begin(), p.end());
    for(std::size_t i = 0; i < n; i++)
    {
        verify_reduced(p, i, options, tolerance);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
