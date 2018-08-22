#include <migraph/gpu/target.hpp>
#include <migraph/gpu/lowering.hpp>
#include <migraph/gpu/write_literals.hpp>
#include <migraph/gpu/context.hpp>
#include <migraph/gpu/eliminate_workspace.hpp>
#include <migraph/gpu/eliminate_allocation.hpp>
#include <migraph/check_context.hpp>
#include <migraph/auto_contiguous.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/simplify_reshapes.hpp>
#include <migraph/eliminate_contiguous.hpp>

namespace migraph {
namespace gpu {

std::vector<pass> target::get_passes(migraph::context& gctx) const
{
    auto& ctx = any_cast<context>(gctx);
    // clang-format off
    return
    {
        dead_code_elimination{},
        auto_contiguous{},
        simplify_reshapes{},
        dead_code_elimination{},
        lowering{ctx},
        eliminate_workspace{},
        eliminate_contiguous{},
        dead_code_elimination{},
        write_literals{&ctx},
        eliminate_allocation{},
        check_context<context>{},
        dead_code_elimination{}
    };
    // clang-format on
}

std::string target::name() const { return "miopen"; }

migraph::context target::get_context() const
{
    return context{share(make_obj<miopen_handle>(&miopenCreate)),
                   share(create_rocblas_handle_ptr())};
}

} // namespace gpu

} // namespace migraph
