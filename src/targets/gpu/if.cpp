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
    auto cond      = args.front().at<bool>();
    module_ref mod = cond ? mods[0] : mods[1];
    std::unordered_map<std::string, argument> params;
    const auto& out_shapes = mod->get_output_shapes();
    for(std::size_t i = 0; i < out_shapes.size(); ++i)
    {
        std::string name = mod->name() + ":#output_" + std::to_string(i);
        auto&& ps        = mod->get_parameter_shape(name);
        if(ps != shape{})
        {
            params[name] = args.at(i + 1);
        }
    }

    auto results = run(mod, params);
    return results[0];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
