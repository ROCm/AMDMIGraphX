#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_REGISTER_OP(code_object_op);

shape code_object_op::compute_shape(std::vector<shape> inputs) const
{
    std::transform(inputs.begin(), inputs.end(), inputs.begin(), [](const shape& s) {
        return s.normalize_standard();
    });
    auto einputs = expected_inputs;
    std::transform(einputs.begin(), einputs.end(), einputs.begin(), [](const shape& s) {
        return s.normalize_standard();
    });
    if(einputs != inputs)
        MIGRAPHX_THROW("Input shapes have changed: [" + to_string_range(einputs) + "] -> [" +
                       to_string_range(inputs) + "]");
    return output;
}
argument
code_object_op::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    std::vector<void*> kargs(args.size());
    std::transform(
        args.begin(), args.end(), kargs.begin(), [](const argument& a) { return a.data(); });
    k.launch(ctx.get_stream().get(), global, local, std::move(kargs));
    return args.back();
}
void code_object_op::finalize(context&, const shape&, const std::vector<shape>&)
{
    assert(not code_object.empty());
    k = kernel(code_object, symbol_name);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
