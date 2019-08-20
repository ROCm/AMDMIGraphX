#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/eliminate_workspace.hpp>
#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/gpu/fuse_ops.hpp>
#include <migraphx/check_context.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/common_subexpression_elimination.hpp>
#include <migraphx/fwd_conv_batchnorm_rewrite.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/gpu/concat_gpu_opt.hpp>
#include <migraphx/gpu/schedule_model.hpp>
#include <migraphx/gpu/adjust_allocation.hpp>
#include <migraphx/gpu/pack_int8_args.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/schedule.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_SCHEDULE_PASS)

std::vector<pass> target::get_passes(migraphx::context& gctx) const
{
    auto& ctx = any_cast<context>(gctx);
    // clang-format off
    return
    {
        dead_code_elimination{},
        simplify_reshapes{},
        dead_code_elimination{},
        eliminate_identity{},
        eliminate_pad{},
        dead_code_elimination{},
        fwd_conv_batchnorm_rewrite{},
        dead_code_elimination{},
        rewrite_rnn{},
        rewrite_pooling{},
        dead_code_elimination{},
        //common_subexpression_elimination{},
        //dead_code_elimination{},
        simplify_algebra{},
        dead_code_elimination{},
        auto_contiguous{},
        simplify_reshapes{},
        dead_code_elimination{},
        propagate_constant{},
        dead_code_elimination{},
        lowering{ctx},
        eliminate_concat{concat_gpu_optimization{}},
        dead_code_elimination{},
        eliminate_contiguous{},
        dead_code_elimination{},
        adjust_allocation{},
        dead_code_elimination{},
        pack_int8_args{},
        dead_code_elimination{},
        fuse_ops{&ctx},
        dead_code_elimination{},
        write_literals{&ctx},
        schedule{gpu::schedule_model{ctx.get_current_device().nstreams()}, enabled(MIGRAPHX_ENABLE_SCHEDULE_PASS{})},
        memory_coloring{"hip::allocate"},
        dead_code_elimination{},
        eliminate_workspace{},
        eliminate_allocation{"hip::allocate"},
        check_context<context>{},
        dead_code_elimination{},
        eliminate_identity{}
    };
    // clang-format on
}

std::string target::name() const { return "miopen"; }

migraphx::context target::get_context() const { return context{}; }
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
