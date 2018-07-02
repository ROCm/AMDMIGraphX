#ifndef MIGRAPH_GUARD_MIGRAPHLIB_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_TARGET_HPP

#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraph/context.hpp>

namespace migraph {

struct program;

<%
interface('target',
    virtual('name', returns='std::string', const=True),
    virtual('apply', returns='void', p='program &', const=True),
    virtual('get_context', returns='context', const=True)
)
%>

} // namespace migraph

#endif
