#ifndef RTG_GUARD_RTGLIB_TARGET_HPP
#define RTG_GUARD_RTGLIB_TARGET_HPP

#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace rtg {

struct program;

<%
interface('target',
    virtual('name', returns='std::string', const=True),
    virtual('apply', returns='void', p='program &', const=True)
)
%>

} // namespace rtg

#endif
