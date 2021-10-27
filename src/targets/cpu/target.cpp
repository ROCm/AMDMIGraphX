
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/check_context.hpp>
#include <migraphx/adjust_allocation.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/eliminate_data_type.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/rewrite_batchnorm.hpp>
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/schedule.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/preallocate_param.hpp>
#include <migraphx/cpu/fuse_ops.hpp>
#include <migraphx/cpu/write_literals.hpp>
#include <migraphx/cpu/allocation_model.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/cpu/lowering.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/normalize_ops.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

std::string target::name() const { return "cpu"; }

// cppcheck-suppress constParameter
std::vector<pass> target::get_passes(migraphx::context& gctx, const compile_options&) const
{
    auto& ctx = any_cast<context>(gctx);
    std::set<shape::type_t> unsupported_types(shape::types().begin(), shape::types().end());
    unsupported_types.erase(shape::type_t::float_type);
    return {normalize_ops{},
            rewrite_quantization{},
            dead_code_elimination{},
            eliminate_data_type{unsupported_types, shape::type_t::float_type},
            dead_code_elimination{},
            simplify_reshapes{},
            eliminate_identity{},
            eliminate_pad{},
            dead_code_elimination{},
            rewrite_batchnorm{},
            dead_code_elimination{},
            rewrite_rnn{},
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
            eliminate_contiguous{"dnnl::reorder"},
            dead_code_elimination{},
            adjust_allocation{cpu_allocation_model{}},
            dead_code_elimination{},
            fuse_ops{&ctx},
            dead_code_elimination{},
            write_literals{},
            dead_code_elimination{},
            memory_coloring{"cpu::allocate"},
            dead_code_elimination{},
            preallocate_param{"scratch", cpu_allocation_model{}},
            dead_code_elimination{}};
}

argument target::allocate(const shape& s) const { return fill_argument(s, 0); }

MIGRAPHX_REGISTER_TARGET(target);

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
