#include <migraphx/gpu/if.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_if::compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
{
    return op.compute_shape({inputs.at(0)}, std::move(mods));
}

argument hip_if::compute(
    const std::vector<argument>& args,
    const std::vector<module_ref>& mods,
    std::function<std::vector<argument>(
        module_ref& mdl, const std::unordered_map<std::string, argument>& inputs)>& run) const
{
    auto cond      = args.at(0).at<bool>();
    module_ref mod = cond ? mods[0] : mods[1];
    std::unordered_map<std::string, argument> params;

    std::size_t out_index = 1;
    for(const auto& smod : mods)
    {
        const auto& param_shapes = smod->get_parameter_shapes();
        for(auto& ns : param_shapes)
        {
            if(contains(ns.first, "#output_"))
            {
                params[ns.first] = args.at(out_index++);
            }
        }
    }

    auto results = run(mod, params);
    return results[0];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
