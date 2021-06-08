#include <iterator>
#include <migraphx/gpu/loop.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_loop::compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
{
    auto offset = inputs.size() - mods.front()->get_output_shapes().size() + 1;
    inputs.erase(inputs.begin() + offset, inputs.end());
    return op.compute_shape(inputs, mods);
}

static std::pair<int, bool> get_name_index(const std::string& name, const std::string& param_prefix)
{
    auto loc = name.find(param_prefix);
    if(loc != std::string::npos)
    {
        int index = std::stoi(name.substr(param_prefix.size()));
        return {index, true};
    }

    std::string out_prefix = "#output_";
    loc                    = name.find(out_prefix)
    {
        int index = std::stoi(name.substr(out_prefix.size()));
        return {index, false};
    }

    return {-1, false};
}

argument
hip_loop::compute(const shape& output_shape,
                  const std::vector<argument>& args,
                  const std::vector<module_ref>& mods,
                  const std::function<std::vector<argument>(
                      module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
{
    auto iter_num            = args.at(0).at<int64_t>();
    auto cond                = args.at(1).at<bool>();
    module_ref mod           = mods.at(0);
    auto mod_out_num         = mod->get_output_shapes().size();
    auto input_num           = args.size() - mod_out_num;
    auto dep_num             = input_num - 2;
    auto param_name_shapes   = mod->get_parameter_shapes();
    std::string param_prefix = "#" + mod->name() + "_in_";

    std::vector<argument> in_args(args.begin() + 2, args.begin() + args.size() - input_num);
    std::vector<argument> out_args(args.begin() + input_num, args.end());

    shape s_iter{shape::int64_type};
    shape s_cond{shape::bool_type};
    int64_t* iter_ptr{};
    hipMalloc((void**)&iter_ptr, sizeof(int64_t));
    bool* cond_ptr{};
    hipMalloc((void**)&cond_ptr, sizeof(bool));
    // insert iter and cond as the first and second elements
    in_args.insert(in_args.begin(), {args.at(1).get_shape(), cond_ptr});
    in_args.insert(in_args.begin(), {args.at(0).get_shape(), iter_ptr});

    for(int64_t iter = 0; (iter < iter_num) and cond; ++iter)
    {
        // copy iter num and cond to device memory
        hipMemcpy(iter_ptr, &iter, sizeof(int64_t), hipMemcpyHostToDevice);
        hipMemcpy(cond_ptr, &cond, sizeof(bool), hipMemcpyHostToDevice);

        // wrap up the inputs and outputs
        std::cout << "loop_compute1" << std::endl;

        std::unordered_map<std::string, argument> params;
        for(auto pn : param_name_shapes)
        {
            auto name     = pn.first;
            auto io_index = get_name_index(name, param_prefix);
            assert(io_index.first != -1);
            // input
            if(io_index.second)
            {
                params[name] = in_args.at(io_index.first).load(pn.second, );
            }
            else
            {
                if(io_index.first >= 1 + dep_num)
                {
                    const auto& arg = in_args.at(io_index.first);
                    params[name]    = arg.load(pn.second, arg.data() + iter * pn.second.bytes());
                }
                else
                {
                    params[name] = out_args.at(io_index.first);
                }
            }
        }

        std::cout << "loop_compute6" << std::endl;
        auto mod_args = run(mod, params);

        std::cout << "loop_compute1" << std::endl;
        // copy back cond to be used next iteration
        hipMemcpy(&cond, mod_args.at(0).data(), sizeof(bool), hipMemcpyDeviceToHost);
        std::copy(mod_args.begin(), mod_args.begin(), in_args.begin() + 1);
        std::cout << "loop_compute7" << std::endl;
    }

    std::cout << "loop_compute8" << std::endl;
    out_args.erase(out_args.begin());

    return argument(out_args);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
