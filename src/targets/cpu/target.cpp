
#include <migraphx/cpu/target.hpp>
#include <migraphx/cpu/lowering.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/dead_code_elimination.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

std::string target::name() const { return "cpu"; }

std::vector<pass> target::get_passes(migraphx::context&) const
{
    return {rewrite_rnn{},
            dead_code_elimination{},
            auto_contiguous{},
            dead_code_elimination{},
            lowering{},
            dead_code_elimination{}};
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
