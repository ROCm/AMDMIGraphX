#include <migraph/gpu/target.hpp>
#include <migraph/gpu/lowering.hpp>
#include <migraph/memory_coloring.hpp>
#include <migraph/gpu/lowering_memory_coloring.hpp>
#include <migraph/gpu/write_literals.hpp>
#include <migraph/gpu/context.hpp>
#include <migraph/gpu/eliminate_workspace.hpp>
#include <migraph/eliminate_allocation.hpp>
#include <migraph/gpu/fuse_ops.hpp>
#include <migraph/check_context.hpp>
#include <migraph/auto_contiguous.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/simplify_reshapes.hpp>
#include <migraph/eliminate_contiguous.hpp>
#include <migraph/fwd_conv_batchnorm_rewrite.hpp>

namespace migraph {
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
        auto_contiguous{},
        simplify_reshapes{},
        dead_code_elimination{},
        lowering{ctx},
        memory_coloring{},
        lowering_memory_coloring{&ctx},
        fuse_ops{},
        dead_code_elimination{},
        eliminate_workspace{},
        eliminate_contiguous{},
        dead_code_elimination{},
        write_literals{&ctx},
        eliminate_allocation{""},
        check_context<context>{},
        dead_code_elimination{}
    };
    // clang-format on
}

std::string target::name() const { return "miopen"; }

migraph::context target::get_context(parameter_map params) const
{
    return context{share(make_obj<miopen_handle>(&miopenCreate)),
                   share(create_rocblas_handle_ptr()),
                   params,
                   {}};
}
} // namespace gpu
} // namespace migraph
