#include <migraphx/adjust_allocation.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/check_context.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/decompose.hpp>
#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/eliminate_data_type.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/inline_module.hpp>
#include <migraphx/insert_pad.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/preallocate_param.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/remap.hpp>
#include <migraphx/rewrite_batchnorm.hpp>
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/schedule.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/gpu/allocation_model.hpp>
#include <migraphx/gpu/concat_gpu_opt.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/eliminate_workspace.hpp>
#include <migraphx/gpu/fuse_ops.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/mlir_conv.hpp>
#include <migraphx/gpu/pack_int8_args.hpp>
#include <migraphx/gpu/schedule_model.hpp>
#include <migraphx/gpu/sync_device.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/write_literals.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_SCHEDULE_PASS)

std::vector<pass> target::get_passes(migraphx::context& gctx, const compile_options& options) const
{
    auto& ctx = any_cast<context>(gctx);
    std::set<shape::type_t> unsupported_types(shape::types().begin(), shape::types().end());
    unsupported_types.erase(shape::type_t::float_type);
    unsupported_types.erase(shape::type_t::half_type);
    unsupported_types.erase(shape::type_t::bool_type);
    unsupported_types.erase(shape::type_t::int8_type);
    unsupported_types.erase(shape::type_t::uint8_type);
    unsupported_types.erase(shape::type_t::tuple_type);
    // clang-format off
    return
    {
        normalize_ops{},
        decompose{},
        dead_code_elimination{},
        rewrite_quantization{},
        dead_code_elimination{},
        eliminate_data_type{unsupported_types, shape::type_t::float_type},
        simplify_reshapes{},
        eliminate_identity{},
        eliminate_pad{},
        dead_code_elimination{},
        insert_pad{},
        dead_code_elimination{},
        rewrite_batchnorm{},
        dead_code_elimination{},
        rewrite_rnn{},
        dead_code_elimination{},
        inline_module{},
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
        mlir_conv{&ctx},
        lowering{&ctx, options.offload_copy},
        eliminate_contiguous{"gpu::contiguous"},
        dead_code_elimination{},
        eliminate_concat{concat_gpu_optimization{}},
        dead_code_elimination{},
        adjust_allocation{gpu_allocation_model{}},
        dead_code_elimination{},
        pack_int8_args{},
        dead_code_elimination{},
        fuse_ops{&ctx, options.fast_math},
        dead_code_elimination{},
        write_literals{&ctx},
        schedule{gpu::schedule_model{ctx.get_current_device().nstreams()}, not enabled(MIGRAPHX_DISABLE_SCHEDULE_PASS{})},
        memory_coloring{"hip::allocate"},
        sync_device{},
        preallocate_param{"scratch", gpu_allocation_model{}},
        dead_code_elimination{},
        eliminate_workspace{},
        eliminate_allocation{"hip::allocate"},
        check_context<context>{},
        normalize_ops{},
        dead_code_elimination{},
        eliminate_identity{}
    };
    // clang-format on
}

std::string target::name() const { return "gpu"; }

migraphx::context target::get_context() const { return context{}; }

argument target::copy_to(const argument& arg) const { return gpu::to_gpu(arg); }

argument target::copy_from(const argument& arg) const { return gpu::from_gpu(arg); }

argument target::allocate(const shape& s) const { return gpu::allocate_gpu(s); }

MIGRAPHX_REGISTER_TARGET(target);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
