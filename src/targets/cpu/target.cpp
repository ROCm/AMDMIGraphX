
#include <migraphx/cpu/target.hpp>
#include <migraphx/cpu/lowering.hpp>
#include <migraphx/auto_contiguous.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace cpu {

std::string target::name() const { return "cpu"; }

std::vector<pass> target::get_passes(migraphx::context&) const
{
    return {auto_contiguous{}, lowering{}};
}

} // namespace cpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
