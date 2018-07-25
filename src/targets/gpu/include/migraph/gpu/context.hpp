#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/rocblas.hpp>

namespace migraph {
namespace gpu {

struct context
{
    shared<miopen_handle> handle;
    shared<rocblas_handle_ptr> rbhandle;
};

} // namespace gpu

} // namespace migraph

#endif
