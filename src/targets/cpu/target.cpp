
#include <migraph/cpu/target.hpp>
#include <migraph/cpu/lowering.hpp>
#include <migraph/auto_contiguous.hpp>

namespace migraph {
namespace MIGRAPH_INLINE_NS {
namespace cpu {

std::string target::name() const { return "cpu"; }

std::vector<pass> target::get_passes(migraph::context&) const
{
    return {auto_contiguous{}, lowering{}};
}

} // namespace cpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraph
