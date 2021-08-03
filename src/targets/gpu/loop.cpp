#include <iterator>
#include <migraphx/gpu/loop.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_loop::compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
{
    inputs.pop_back();
    inputs.pop_back();
    inputs.erase(inputs.begin() + 3);
    inputs.erase(inputs.begin() + 1);
    return op.compute_shape(inputs, mods);
}

static std::pair<int, bool> get_name_index(const std::string& name, const std::string& param_prefix)
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

argument
hip_loop::compute(context& ctx,
                  const shape&,
                  const std::vector<argument>& args,
                  const std::vector<module_ref>& mods,
                  const std::function<std::vector<argument>(
                      module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
{
    auto cpy_args = args;
    const auto arg_iter_num = cpy_args.at(1);
    const auto arg_cond = cpy_args.at(3);
    cpy_args.erase(cpy_args.begin() + 3);
    cpy_args.erase(cpy_args.begin() + 1);

    auto iter_num = arg_iter_num.at<int64_t>();
    if(iter_num > op.max_iter_num)
    {
        MIGRAPHX_THROW("LOOP compute(): iter_num " + std::to_string(iter_num) +
                       " larger than max_iter_num " + std::to_string(op.max_iter_num) +
                       ", please set the max_iter_num to at least " + std::to_string(iter_num) +
                       " in onnx_options before parsing onnx file!");
    }

    auto cond                = arg_cond.at<bool>();
    module_ref mod           = mods.at(0);
    auto input_num           = cpy_args.size() - 2;
    auto dep_num             = input_num - 2;
    auto param_name_shapes   = mod->get_parameter_shapes();
    std::string param_prefix = "#" + mod->name() + "_in_";
    std::vector<argument> in_args(cpy_args.begin(), cpy_args.begin() + input_num);
    std::vector<argument> out_args = {cpy_args.at(input_num)};
    auto mod_outputs               = cpy_args.back().get_sub_objects();
    out_args.insert(out_args.end(), mod_outputs.begin(), mod_outputs.end());

    int64_t iter = 0;
    for(iter = 0; (iter < iter_num) and cond; ++iter)
    {
        // copy iter num and cond to device memory
        (void)hipMemcpy(in_args.at(0).data(), &iter, sizeof(int64_t), hipMemcpyHostToDevice);
        (void)hipMemcpy(in_args.at(1).data(), &cond, sizeof(bool), hipMemcpyHostToDevice);

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
        (void)hipMemcpy(&cond, mod_args.at(0).data(), sizeof(bool), hipMemcpyDeviceToHost);
        std::copy(mod_args.begin(), mod_args.begin() + dep_num + 1, in_args.begin() + 1);
        std::copy(mod_args.begin(), mod_args.begin() + dep_num + 1, out_args.begin());
    }

    out_args.erase(out_args.begin());
    std::vector<argument> scan_outputs(out_args.begin() + dep_num, out_args.end());

    // set unused scan outputs to 0s
    if(iter < iter_num)
    {
        auto elem_num = iter_num - iter;
        for(auto& out : std::vector<argument>(out_args.begin() + dep_num, out_args.end()))
        {
            auto s    = out.get_shape();
            auto size = s.bytes() / iter_num;
            (void)hipMemset(out.data() + iter * size, 0, size * elem_num);
        }
    }

    return argument(out_args);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
