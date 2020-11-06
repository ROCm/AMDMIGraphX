
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/check_context.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/decompose.hpp>
#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/remap.hpp>
#include <migraphx/rewrite_batchnorm.hpp>
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/schedule.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/cpu/lowering.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

std::string target::name() const { return "cpu"; }

std::vector<pass> target::get_passes(migraphx::context&, const compile_options&) const
{
    return {rewrite_rnn{},
            dead_code_elimination{},
            decompose{},
            dead_code_elimination{},
            simplify_reshapes{},
            eliminate_identity{},
            eliminate_pad{},
            dead_code_elimination{},
            rewrite_batchnorm{},
            dead_code_elimination{},
            rewrite_pooling{},
            dead_code_elimination{},
            eliminate_common_subexpression{},
            dead_code_elimination{},
            simplify_algebra{},
            simplify_reshapes{},
            simplify_algebra{},
            auto_contiguous{},
            simplify_reshapes{},
            propagate_constant{},
            dead_code_elimination{},
            lowering{},
            dead_code_elimination{}};
}

argument target::allocate(const shape& s) const { return fill_argument(s, 0); }

MIGRAPHX_REGISTER_TARGET(target);

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
