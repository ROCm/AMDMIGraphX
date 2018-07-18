#include <migraph/miopen/miopen_target.hpp>
#include <migraph/miopen/lowering.hpp>
#include <migraph/miopen/miopen_write_literals.hpp>
#include <migraph/miopen/context.hpp>

namespace migraph {
namespace miopen {

std::vector<pass> miopen_target::get_passes(context&) const
{
    return {lowering{}, miopen_write_literals{}};
}

std::string miopen_target::name() const { return "miopen"; }

context miopen_target::get_context() const
{
    return miopen_context{share(make_obj<miopen_handle>(&miopenCreate)),
                          share(create_rocblas_handle_ptr())};
}

} // namespace miopen

} // namespace migraph
