#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

#include <migraph/miopen/miopen.hpp>

namespace migraph {
namespace miopen {

struct miopen_context
{
    shared<miopen_handle> handle;
};

} // namespace miopen

} // namespace migraph

#endif
