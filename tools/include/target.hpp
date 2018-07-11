#ifndef MIGRAPH_GUARD_MIGRAPHLIB_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_TARGET_HPP

#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <migraph/context.hpp>
#include <migraph/pass.hpp>

namespace migraph {

struct program;

<%
interface('target',
    virtual('name', returns='std::string', const=True),
    virtual('get_passes', ctx='context&', returns='std::vector<pass>', const=True),
    virtual('get_context', returns='context', const=True)
)
%>

} // namespace migraph

#endif
