#include <migraphx/cpu/allocation_model.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

std::string cpu_allocation_model::name() const { return "cpu::allocate"; }
operation cpu_allocation_model::allocate(const shape& s) const
{
    return make_op(name(), {{"shape", to_value(s)}});
}

operation cpu_allocation_model::preallocate(const shape& s, const std::string& id) const
{
    return make_op("cpu::preallocate", {{"shape", to_value(s)}, {"id", id}});
}

std::string cpu_allocation_model::copy() const { return "cpu::copy"; }

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
