#ifndef MIGRAPH_GUARD_MIGRAPHLIB_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_TARGET_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <migraph/context.hpp>
#include <migraph/pass.hpp>

namespace migraph {

#ifdef DOXYGEN

/// An interface for a compilation target
struct target
{
    /// A unique name used to identify the target
    std::string name() const;
    /// The transformation passes to be run
    /**
     * @brief The transformation pass to be run during compilation.
     * @details [long description]
     *
     * @param ctx This is the target-dependent context that is created by `get_context`
     * @return The passes to be ran
     */
    std::vector<pass> get_passes(context& ctx) const;
    /**
     * @brief Construct a context for the target.
     * @return The context to be used during compilation and execution.
     */
    context get_context() const;
};

#else

<%
interface('target',
    virtual('name', returns='std::string', const=True),
    virtual('get_passes', ctx='context&', returns='std::vector<pass>', const=True),
    virtual('get_context', returns='context', const=True)
)
%>

#endif

} // namespace migraph

#endif
