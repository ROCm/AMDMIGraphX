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
    auto get_output_index = [](const std::string& name)
    {
        std::string out_prefix = "#output_";
        auto loc                    = name.find(out_prefix);
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
    auto cond     = args.at(2).at<bool>();

    args.erase(args.begin() + 2);
    args.erase(args.begin());

    auto input_num = args.size() - 2;
    auto dep_num   = input_num - 2;

    module_ref mod           = mods.at(0);
    auto param_name_shapes   = mod->get_parameter_shapes();
    auto param_names         = mod->get_parameter_names();
    // std::string param_prefix = "#" + mod->name() + "_in_";

    std::vector<argument> in_args(args.begin(), args.begin() + input_num);
    std::vector<argument> out_args = {args.at(input_num)};
    auto mod_outputs               = args.back().get_sub_objects();
    out_args.insert(out_args.end(), mod_outputs.begin(), mod_outputs.end());

    std::vector<argument> scan_outputs(out_args.begin() + dep_num + 1, out_args.end());

    int64_t iter = 0;
    for(iter = 0; iter < iter_num and cond; ++iter)
    {
        // copy iter num and cond to device memory
        model.copy(ctx, iter, in_args.at(0));
        model.copy(ctx, cond, in_args.at(1));

        // wrap up the inputs and outputs
        std::unordered_map<std::string, argument> params;
        int input_index = 0;
        for(const auto& name : param_names)
        {
            auto output_index = get_output_index(name);
            // it is an input parameter
            if (output_index == -1)
            {
                params[name] = in_args.at(input_index++);
            }
            else
            {
                auto out_s = mod->get_parameter_shape(name);
                if(output_index > dep_num)
                {
                    const auto& arg = out_args.at(output_index);
                    assert((iter + 1) * out_s.bytes() <= arg.get_shape().bytes());
                    params[name] = argument::load(out_s, arg.data() + iter * out_s.bytes());
                }
                else
                {
                    params[name] = out_args.at(output_index);
                }
            }
        }

        auto mod_args = run(mod, params);
        ctx.finish();

        // copy back cond to be used next iteration
        model.copy(ctx, mod_args.at(0), cond);

        // std::copy(mod_args.begin(), mod_args.begin() + dep_num + 1, in_args.begin() + 1);
        model.copy_carry_dependencies(ctx, mod_args.begin(), mod_args.begin() + dep_num + 1, in_args.begin() + 1);

        std::vector<argument> mod_scan_outs(mod_args.begin() + 1 + dep_num, mod_args.end());
        model.append(mod_scan_outs, scan_outputs, iter);
        ctx.finish();
    }

    out_args.erase(out_args.begin());
    std::copy(in_args.begin() + 2, in_args.end(), out_args.begin());
    model.set_zero(ctx, scan_outputs, iter);

    return argument(out_args);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
