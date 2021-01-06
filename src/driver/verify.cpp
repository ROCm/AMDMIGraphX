#include "verify.hpp"
#include "perf.hpp"

#include <migraphx/ref/target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify_args.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/compile_options.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<argument> run_ref(program p, const parameter_map& inputs)
{
    p.compile(ref::target{});
    auto out = p.eval(inputs);
    std::cout << p << std::endl;
    return out;
}

std::vector<argument>
run_target(program p, const target& t, const compile_options& options, const parameter_map& inputs)
{
    p.compile(t, options);

    parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        auto arg   = inputs.count(x.first) == 0 ? generate_argument(x.second) : inputs.at(x.first);
        m[x.first] = options.offload_copy ? arg : t.copy_to(arg);
    }
    auto gpu_out = p.eval(m);
    std::vector<argument> output(gpu_out.size());
    std::cout << p << std::endl;
    std::transform(gpu_out.begin(), gpu_out.end(), output.begin(), [&](auto& argu) {
        return options.offload_copy ? argu : t.copy_from(argu);
    });
    return output;
}

void verify_program(const std::string& name,
                    const program& p,
                    const target& t,
                    compile_options options,
                    const parameter_map& inputs,
                    double tolerance)
{
    auto x = run_ref(p, inputs);
    auto y = run_target(p, t, options, inputs);

    std::size_t output_num = x.size();
    for(std::size_t i = 0; i < output_num; ++i)
    {
        verify_args(name, x[i], y[i], tolerance);
    }
    // std::cout << "cpu: " << x << std::endl;
    // std::cout << "gpu: " << y << std::endl;
}

void verify_instructions(const program& prog,
                         const target& t,
                         compile_options options,
                         double tolerance)
{
    const auto* mm_prog = prog.get_main_module();
    for(auto&& ins : (*mm_prog))
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
        auto* mm_p = p.get_main_module();
        std::vector<instruction_ref> inputs;
        for(auto&& arg : ins.inputs())
        {
            if(arg->name() == "@literal")
                inputs.push_back(mm_p->add_literal(arg->get_literal()));
            else
                inputs.push_back(
                    mm_p->add_parameter(std::to_string(inputs.size()), arg->get_shape()));
        }
        mm_p->add_instruction(ins.get_operator(), inputs);
        try
        {
            std::cout << "Verify: " << ins.name() << std::endl;
            std::cout << p << std::endl;
            verify_program(ins.name(), p, t, options, create_param_map(p, false), tolerance);
        }
        catch(...)
        {
            std::cout << "Instruction " << ins.name() << " threw an exception." << std::endl;
            throw;
        }
    }
}

void verify_reduced(program p,
                    int n,
                    const target& t,
                    compile_options options,
                    const parameter_map& inputs,
                    double tolerance)
{
    auto* mm  = p.get_main_module();
    auto last = std::prev(mm->end(), n + 1);
    mm->remove_instructions(last, mm->end());
    std::cout << "Verify: " << std::endl;
    std::cout << p << std::endl;
    verify_program(std::to_string(n), p, t, options, inputs, tolerance);
}

void verify_reduced_program(const program& p,
                            const target& t,
                            compile_options options,
                            const parameter_map& inputs,
                            double tolerance)
{
    const auto* mm = p.get_main_module();
    auto n         = std::distance(mm->begin(), mm->end());
    for(std::size_t i = 0; i < n; i++)
    {
        verify_reduced(p, i, t, options, inputs, tolerance);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
