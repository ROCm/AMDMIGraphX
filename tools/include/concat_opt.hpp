#ifndef MIGRAPHX_GUARD_CONCAT_OPT_HPP
#define MIGRAPHX_GUARD_CONCAT_OPT_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/operation.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

#ifdef DOXYGEN

/// An interface for target-dependent optimization for the concat instruction
struct concat_optimization
{
    /// The name of the target-dependent concat operator
    std::string name() const;
    /// A name of the target-dependent allocate operator
    std::string allocate() const;
    /// Return the target-independent concat operator
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

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
