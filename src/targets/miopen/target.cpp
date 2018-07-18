#include <migraph/miopen/target.hpp>
#include <migraph/miopen/lowering.hpp>
#include <migraph/miopen/write_literals.hpp>
#include <migraph/miopen/context.hpp>

namespace migraph {
namespace miopen {

std::vector<pass> target::get_passes(migraph::context&) const { return {lowering{}, write_literals{}}; }

std::string target::name() const { return "miopen"; }

migraph::context target::get_context() const
{
    return context{share(make_obj<miopen_handle>(&miopenCreate)),
                          share(create_rocblas_handle_ptr())};
}

} // namespace miopen

} // namespace migraph
