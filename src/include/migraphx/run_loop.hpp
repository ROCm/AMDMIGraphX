#ifndef MIGRAPHX_GUARD_RTGLIB_RUN_LOOP_HPP
#define MIGRAPHX_GUARD_RTGLIB_RUN_LOOP_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/context.hpp>
#include <migraphx/module.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class LoopModel, class T>
argument run_loop(const LoopModel& model,
                  T& ctx,
                  std::vector<argument> args,
                  const std::vector<module_ref>& mods,
                  const std::function<std::vector<argument>(
                      module_ref&, const std::unordered_map<std::string, argument>&)>& run)
{
    auto get_output_index = [](const std::string& name) {
        std::string out_prefix = "#output_";
        auto loc               = name.find(out_prefix);
        if(loc != std::string::npos)
        {
            int index = std::stoi(name.substr(loc + out_prefix.size()));
            return index;
        }

        return -1;
    };

    std::vector<std::vector<argument>> results;
    // process argu lists
    auto iter_num = args.at(0).at<int64_t>();
    auto cond     = args.at(1).at<bool>();

    auto input_num = (args.size() - 2) / 2;
    auto dep_num   = input_num - 2;

    module_ref mod         = mods.at(0);
    auto param_name_shapes = mod->get_parameter_shapes();
    auto param_names       = mod->get_parameter_names();

    std::vector<argument> dep0(args.begin() + input_num + 1, args.begin() + 2 * input_num);
    std::vector<argument> dep1(args.begin() + 2 * input_num, args.begin() + 2 * input_num + 1);
    auto ins_outputs = args.back().get_sub_objects();
    dep1.insert(dep1.end(), ins_outputs.begin(), ins_outputs.begin() + dep_num);
    std::array<std::vector<argument>, 2> loop_carry_deps = {dep0, dep1};

    // loop iter argument
    std::vector<argument> in_args = {args.at(input_num), dep1.at(0)};
    in_args.insert(in_args.end(), args.begin() + 2, args.begin() + input_num);

    std::vector<argument> out_args = dep0;
    out_args.insert(out_args.end(), ins_outputs.begin() + dep_num, ins_outputs.end());
    std::vector<argument> scan_outputs(ins_outputs.begin() + dep_num, ins_outputs.end());

    int64_t iter = 0;
    for(iter = 0; iter < iter_num and cond; ++iter)
    {
        std::cout << "1.iter = " << iter << ", cond = " << cond << std::endl;
        // copy iter num and cond to device memory
        model.copy(ctx, iter, in_args.at(0));
        model.copy(ctx, cond, in_args.at(1));

        // wrap up the inputs and outputs
        std::unordered_map<std::string, argument> params;
        int input_index = 0;
        for(const auto& name : param_names)
        {
            auto ps = mod->get_parameter_shape(name);
            if(ps == shape{})
            {
                continue;
            }

            auto output_index = get_output_index(name);
            // it is an input parameter
            if(output_index == -1)
            {
                params[name] = in_args.at(input_index++);
            }
            else
            {
                if(output_index > dep_num)
                {
                    const auto& arg = out_args.at(output_index);
                    assert((iter + 1) * ps.bytes() <= arg.get_shape().bytes());
                    params[name] = argument::load(ps, arg.data() + iter * ps.bytes());
                }
                else
                {
                    params[name] = out_args.at(output_index);
                }
            }
        }

        model.print_params(params);

        auto mod_args = run(mod, params);
        ctx.finish();

        model.print_outputs(mod_args);

        // copy back cond to be used next iteration
        model.copy(ctx, mod_args.at(0), cond);
        ctx.finish();

        std::cout << "2.iter = " << iter << ", cond = " << cond << std::endl;

        // mod outputs are used as next loop input
        std::copy(mod_args.begin(), mod_args.begin() + dep_num + 1, in_args.begin() + 1);
        const auto& dep_out = loop_carry_deps[(iter + 1) & 1];
        std::copy(dep_out.begin(), dep_out.end(), out_args.begin());

        std::vector<argument> mod_scan_outs(mod_args.begin() + 1 + dep_num, mod_args.end());
        model.append(mod_scan_outs, scan_outputs, iter);

        std::cout << "3.iter = " << iter << ", cond = " << cond << std::endl;
    }

    out_args.erase(out_args.begin());
    std::copy(in_args.begin() + 2, in_args.end(), out_args.begin());
    model.set_zero(ctx, scan_outputs, iter);

    return argument(out_args);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
