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

std::string gpu_allocation_model::copy() const { return "hip::copy"; }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
