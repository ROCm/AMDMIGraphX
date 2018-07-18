#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

#include <migraph/miopen/miopen.hpp>
#include <migraph/miopen/rocblas.hpp>

namespace migraph {
namespace miopen {

struct context
{
    shared<miopen_handle> handle;
    shared<rocblas_handle_ptr> rbhandle;
};

} // namespace miopen

} // namespace migraph

#endif
