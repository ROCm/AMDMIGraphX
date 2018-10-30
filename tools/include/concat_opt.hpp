#ifndef MIGRAPH_GUARD_CONCAT_OPT_HPP
#define MIGRAPH_GUARD_CONCAT_OPT_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraph/operation.hpp>
#include <migraph/operators.hpp>

namespace migraph {

struct program;

#ifdef DOXYGEN

/// An interface for applying an optimization for the concat instruction
struct concat_optimization
{
    /// A unique name used to identify the concat optimization
    std::string name() const;
    /// A unique name used to identify the allocate operator
    std::string allocate() const;
    /// Return the lowered concat operator
    op::concat get_concat(const operation& op) const;
};

#else

<%
interface('concat_optimization',
    virtual('name', returns='std::string', const=True),
    virtual('allocate', returns='std::string', const=True),
    virtual('get_concat', returns='op::concat', op='const operation&', const=True)
)
%>

#endif

} // namespace migraph

#endif
