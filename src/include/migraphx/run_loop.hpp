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

static std::pair<int, bool> get_name_index(const std::string& name,
                                           const std::string& param_prefix) const
{
    auto loc = name.find(param_prefix);
    if(loc != std::string::npos)
    {
        int index = std::stoi(name.substr(loc + param_prefix.size()));
        return {index, true};
    }

    std::string out_prefix = "#output_";
    loc                    = name.find(out_prefix);
    if(loc != std::string::npos)
    {
        int index = std::stoi(name.substr(loc + out_prefix.size()));
        return {index, false};
    }

    return {-1, false};
}

template <class LoopModel>
std::vector<argument>
run_loop(const LoopModel& model,
         context& ctx,
         std::vector<argument> args,
         const std::vector<module_ref>& mods,
         const std::function<std::vector<argument>(
             module_ref&, const std::unordered_map<std::string, argument>&)>& run)
{
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
    std::string param_prefix = "#" + mod->name() + "_in_";

    std::vector<argument> in_args(args.begin(), args.begin() + input_num);
    std::vector<argument> out_args = {args.at(input_num)};
    auto mod_outputs               = args.back().get_sub_objects();
    out_args.insert(out_args.end(), mod_outputs.begin(), mod_outputs.end());

    std::vector<argument> scan_outputs(out_args.begin() + dep_num + 1, out_args.end());

    int64_t iter = 0;
    for(iter = 0; (iter < iter_num) and cond; ++iter)
    {
        // copy iter num and cond to device memory
        model.copy_arg(in_args.at(0), iter, true);
        model.copy_arg(in_args.at(1), cond, true);
        // (void)hipMemcpy(in_args.at(0).data(), &iter, sizeof(int64_t), hipMemcpyHostToDevice);
        // (void)hipMemcpy(in_args.at(1).data(), &cond, sizeof(bool), hipMemcpyHostToDevice);

        // wrap up the inputs and outputs
        std::unordered_map<std::string, argument> params;
        for(auto pn : param_name_shapes)
        {
            auto name     = pn.first;
            auto io_index = get_name_index(name, param_prefix);
            assert(io_index.first != -1);
            // name is for input
            if(io_index.second)
            {
                params[name] = in_args.at(io_index.first);
            }
            else
            {
                if(io_index.first > dep_num)
                {
                    const auto& arg = out_args.at(io_index.first);
                    params[name]    = arg.load(pn.second, arg.data() + iter * pn.second.bytes());
                }
                else
                {
                    params[name] = out_args.at(io_index.first);
                }
            }
        }

        auto mod_args = run(mod, params);
        ctx.finish();

        // copy back cond to be used next iteration
        model.copy_arg(mod_args.at(0), cond, false);
        std::copy(mod_args.begin(), mod_args.begin() + dep_num + 1, in_args.begin() + 1);

        // concat scan outputs
        std::vector<argument> mod_scan_outs(mod_args.begin() + 1 + dep_num, mod_args.end());
        model.concat_scan_outputs(mod_scan_outs, scan_outputs);
    }

    model.set_zero(scan_outputs, iter_num, iter);

    return
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
