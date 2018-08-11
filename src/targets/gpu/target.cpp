#include <migraph/gpu/target.hpp>
#include <migraph/gpu/lowering.hpp>
#include <migraph/memory_coloring.hpp>
#include <migraph/gpu/write_literals.hpp>
#include <migraph/gpu/context.hpp>

namespace migraph {
namespace gpu {

std::vector<pass> target::get_passes(migraph::context&) const
{
    return {lowering{}, memory_coloring{}, write_literals{}};
}

std::string target::name() const { return "miopen"; }

migraph::context target::get_context() const
{
    return context{share(make_obj<miopen_handle>(&miopenCreate)),
                   share(create_rocblas_handle_ptr())};
}

} // namespace gpu

} // namespace migraph
