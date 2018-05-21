#ifndef RTG_GUARD_RTGLIB_OPERAND_HPP
#define RTG_GUARD_RTGLIB_OPERAND_HPP

#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <rtg/shape.hpp>
#include <rtg/argument.hpp>

namespace rtg {

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
    virtual('compute', returns='argument', output='shape', input='std::vector<argument>', const=True),
    friend('operator<<', returns='std::ostream &', os='std::ostream &', op='const operation &', using='rtg::operation_stream::operator<<')
)
%>

} // namespace rtg

#endif
