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

argument
hip_loop::compute(const shape& output_shape,
                  const std::vector<argument>& args,
                  const std::vector<module_ref>& mods,
                  const std::function<std::vector<argument>(
                      module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
{
    auto iter_num                   = args.at(0).at<int64_t>();
    auto cond                       = args.at(1).at<bool>();
    module_ref mod                  = mods.at(0);
    auto mod_name                   = mod->name();
    std::vector<std::string> pnames = mod->get_parameter_names();

    std::string prefix = "@mgx_" + mod->name();
    std::vector<std::pair<std::string, bool>> fixed_input_pair;
    auto it = std::find_if(
        pnames.begin(), pnames.end(), [&](auto name) { return contains(name, prefix + "_iter_"); });
    if(it != pnames.end())
    {
        fixed_input_pair.push_back({*it, true});
        pnames.erase(it);
    }
    else
    {
        fixed_input_pair.push_back({{}, false});
    }

    it = std::find_if(
        pnames.begin(), pnames.end(), [&](auto name) { return contains(name, prefix + "_cond_"); });
    if(it != pnames.end())
    {
        fixed_input_pair.push_back({*it, true});
        pnames.erase(it);
    }
    else
    {
        fixed_input_pair.push_back({{}, false});
    }

    std::vector<shape> vec_out_shapes = output_shape.sub_shapes();
    std::size_t dep_num               = args.size() - 2 - vec_out_shapes.size();

    // dependency carry outputs
    std::vector<argument> dep_outputs(args.begin() + 2, args.begin() + 2 + dep_num);

    // scan outputs
    std::vector<argument> scan_outputs(args.begin() + dep_num + 2, args.end());

    // sub graph inputs for each iteration
    std::vector<argument> mod_args(args.begin() + 1, args.begin() + 1 + dep_num);
    shape s_iter{shape::int64_type};
    shape s_cond{shape::bool_type};
    int64_t* iter_ptr{};
    hipMalloc((void**)&iter_ptr, sizeof(int64_t));
    bool* cond_ptr{};
    hipMalloc((void**)&cond_ptr, sizeof(bool));

    for(int64_t iter = 0; (iter < iter_num) and cond; ++iter)
    {

        std::cout << "loop_compute1" << std::endl;
        std::unordered_map<std::string, argument> params;

        // iter index
        if(fixed_input_pair.at(0).second)
        {
            hipMemcpy(iter_ptr, &iter, sizeof(int64_t), hipMemcpyHostToDevice);
            params[fixed_input_pair.at(0).first] = argument(s_iter, iter_ptr);
        }
        std::cout << "loop_compute2" << std::endl;

        // cond variable
        if(fixed_input_pair.at(1).second)
        {
            hipMemcpy(cond_ptr, &cond, sizeof(bool), hipMemcpyHostToDevice);
            params[fixed_input_pair.at(1).first] = argument(s_cond, cond_ptr);
        }
        std::cout << "loop_compute3, dep_num = " << dep_num << std::endl;

        // wrapup dependency carry output parameters
        std::transform(
            vec_out_shapes.begin(),
            vec_out_shapes.begin() + dep_num,
            dep_outputs.begin(),
            std::back_inserter(mod_args),
            [&](auto s, auto arg) { return arg.load(s, arg.data() + iter * s.bytes()); });

        std::cout << "loop_compute4, vec_out_size = " << vec_out_shapes.size()
                  << ", scan_out_size = " << scan_outputs.size() << std::endl;
        // wrapup scan output parameters
        std::transform(vec_out_shapes.begin() + dep_num,
                       vec_out_shapes.end(),
                       scan_outputs.begin(),
                       std::back_inserter(mod_args),
                       [&](auto s, auto arg) {
                           std::cout << "s = " << s << ", iter = " << iter << std::endl;
                           return arg.load(s, arg.data() + iter * s.bytes());
                       });

        std::cout << "loop_compute5" << std::endl;
        // carry dependencies
        std::transform(pnames.begin(),
                       pnames.end(),
                       mod_args.begin() + 1,
                       std::inserter(params, params.end()),
                       [](auto&& name, auto&& arg) { return std::make_pair(name, arg); });

        std::cout << "loop_compute6" << std::endl;
        mod_args = run(mod, params);

        std::cout << "loop_compute1" << std::endl;
        // copy back cond to be used next iteration
        hipMemcpy(&cond, mod_args.at(0).data(), sizeof(bool), hipMemcpyDeviceToHost);
        std::cout << "loop_compute7" << std::endl;
    }

    std::cout << "loop_compute8" << std::endl;
    // remove the cond variable
    mod_args.erase(mod_args.begin());
    auto outputs = mod_args;
    outputs.insert(outputs.end(), scan_outputs.begin(), scan_outputs.end());

    return argument(outputs);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
