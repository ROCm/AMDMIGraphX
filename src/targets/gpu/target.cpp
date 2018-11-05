#include <migraph/gpu/target.hpp>
#include <migraph/gpu/lowering.hpp>
#include <migraph/memory_coloring.hpp>
#include <migraph/gpu/write_literals.hpp>
#include <migraph/gpu/context.hpp>
#include <migraph/gpu/eliminate_workspace.hpp>
#include <migraph/eliminate_allocation.hpp>
#include <migraph/gpu/fuse_ops.hpp>
#include <migraph/check_context.hpp>
#include <migraph/auto_contiguous.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/simplify_reshapes.hpp>
#include <migraph/simplify_algebra.hpp>
#include <migraph/constant_propagate.hpp>
#include <migraph/eliminate_contiguous.hpp>
#include <migraph/common_subexpression_elimination.hpp>
#include <migraph/fwd_conv_batchnorm_rewrite.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

std::vector<pass> target::get_passes(migraph::context& gctx) const
{
    auto& ctx = any_cast<context>(gctx);
    // clang-format off
    return
    {
        dead_code_elimination{},
        fwd_conv_batchnorm_rewrite{},
        dead_code_elimination{},
        common_subexpression_elimination{},
        dead_code_elimination{},
        simplify_algebra{},
        dead_code_elimination{},
        constant_propagate{},
        dead_code_elimination{},
        auto_contiguous{},
        simplify_reshapes{},
        dead_code_elimination{},
        lowering{ctx},
        eliminate_contiguous{},
        dead_code_elimination{},
        fuse_ops{&ctx},
        dead_code_elimination{},
        write_literals{&ctx},
        memory_coloring{"hip::allocate"},
        eliminate_workspace{},
        eliminate_allocation{"hip::allocate"},
        check_context<context>{},
        dead_code_elimination{}
    };
    // clang-format on
}

std::string target::name() const { return "miopen"; }

migraph::context target::get_context() const { return context{}; }
} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraph
