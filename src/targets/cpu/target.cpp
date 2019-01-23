
#include <migraphx/cpu/target.hpp>
#include <migraphx/cpu/lowering.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/rewrite_gru.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

std::string target::name() const { return "cpu"; }

std::vector<pass> target::get_passes(migraphx::context&) const
{
    return {auto_contiguous{}, 
        rewrite_rnn{},
        rewrite_gru{},
        lowering{}};
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
