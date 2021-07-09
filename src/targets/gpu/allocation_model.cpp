#include <migraphx/gpu/allocation_model.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

std::string gpu_allocation_model::name() const { return "hip::allocate"; }
operation gpu_allocation_model::allocate(const shape& s) const
{
    return make_op(name(), {{"shape", to_value(s)}});
}

operation gpu_allocation_model::preallocate(const shape& s, const std::string& id) const
{
    return make_op("hip::hip_allocate_memory", {{"shape", to_value(s)}, {"id", id}});
}

std::string gpu_allocation_model::copy() const { return "hip::copy"; }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
