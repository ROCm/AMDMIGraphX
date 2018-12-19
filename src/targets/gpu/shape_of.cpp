#include <migraphx/gpu/shape_of.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_shape::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2);
    return {shape::int64_type, {1, inputs[0].lens().size()}};
}

argument hip_shape::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    std::vector<std::size_t> in_shape = args[0].get_shape().lens();
    std::vector<uint64_t> vec_shape(in_shape.size());
    std::transform(in_shape.begin(), in_shape.end(), vec_shape.begin(), 
            [](auto &i) { return static_cast<uint64_t>(i);}
            );
    argument shape_arg = to_gpu(migraphx::argument{args[1].get_shape(), 
            reinterpret_cast<char *>(&vec_shape[0])});
    device::shape_of(ctx.get_stream().get(), args[1], shape_arg);
    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
