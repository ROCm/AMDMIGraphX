#include <migraphx/gpu/if.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_if::compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
{
    inputs.pop_back();
    return op.compute_shape(inputs, mods);
}

argument hip_if::compute(
    const std::vector<argument>& args,
    const std::vector<module_ref>& mods,
    std::function<std::vector<argument>(
        module_ref& mdl, const std::unordered_map<std::string, argument>& inputs)>& run) const
{
    auto arg_cond  = migraphx::gpu::from_gpu(args[0]);
    auto cond      = arg_cond.at<bool>();
    module_ref mdl = cond ? mods[0] : mods[1];
    auto results   = run(mdl, {});
    context ctx{};
    gpu_copy(ctx, results[0], args.back());

    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
