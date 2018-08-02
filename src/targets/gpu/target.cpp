#include <migraph/gpu/target.hpp>
#include <migraph/gpu/lowering.hpp>
#include <migraph/gpu/write_literals.hpp>
#include <migraph/gpu/context.hpp>
#include <migraph/check_context.hpp>

namespace migraph {
namespace gpu {

std::vector<pass> target::get_passes(migraph::context&) const
{
    return {lowering{}, write_literals{}, check_context<context>{}};
}

std::string target::name() const { return "miopen"; }

migraph::context target::get_context() const
{
    return context{share(make_obj<miopen_handle>(&miopenCreate)),
                   share(create_rocblas_handle_ptr())};
}

} // namespace gpu

} // namespace migraph
