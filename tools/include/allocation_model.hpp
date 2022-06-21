#ifndef MIGRAPHX_GUARD_ALLOCATION_MODEL_HPP
#define MIGRAPHX_GUARD_ALLOCATION_MODEL_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// An interface for target-dependent allocation
struct allocation_model
{
    /// A name of the target-dependent allocate operator
    std::string name() const;
    /// A name of the target-dependent copy operator
    std::string copy() const;
    /// Create an allocation operator for the given shape
    operation allocate(const shape& s) const;
    /// Create a preallocated operator for the given shape
    operation preallocate(const shape& s, const std::string& id) const;
    /// Check if outputs are to be inserted
    bool needs_out_params() const;
};

#else

<%
interface('allocation_model',
    virtual('name', returns='std::string', const=True),
    virtual('copy', returns='std::string', const=True),
    virtual('allocate', s='const shape&', returns='operation', const=True),
    virtual('preallocate', s='const shape&', id='std::string', returns='operation', const=True),
    virtual('needs_out_params', returns='bool', const=True)
)
%>

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
