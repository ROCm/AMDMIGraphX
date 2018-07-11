#ifndef MIGRAPH_GUARD_PASS_HPP
#define MIGRAPH_GUARD_PASS_HPP

#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace migraph {

struct program;

<%
interface('pass',
    virtual('name', returns='std::string', const=True),
    virtual('apply', returns='void', p='program &', const=True)
)
%>

} // namespace migraph

#endif
