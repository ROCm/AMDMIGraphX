#ifndef MIGRAPH_GUARD_MIGRAPHLIB_OPERAND_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_OPERAND_HPP

#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraph/shape.hpp>
#include <migraph/argument.hpp>
#include <migraph/context.hpp>

namespace migraph {

namespace operation_stream {

template <class T>
auto operator<<(std::ostream& os, const T& x) -> decltype(os << x.name())
{
    return os << x.name();
}

} // namespace operation_stream

<%
interface('operation',
    virtual('name', returns='std::string', const=True),
    virtual('compute_shape', returns='shape', input='std::vector<shape>', const=True),
    virtual('compute', returns='argument', ctx='context&', output='shape', input='std::vector<argument>', const=True),
    friend('operator<<', returns='std::ostream &', os='std::ostream &', op='const operation &', using='migraph::operation_stream::operator<<')
)
%>

} // namespace migraph

#endif
