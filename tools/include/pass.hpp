#ifndef MIGRAPH_GUARD_PASS_HPP
#define MIGRAPH_GUARD_PASS_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace migraph {

struct program;

#ifdef DOXYGEN

/// An interface for applying a transformation to the instructions in a
/// `program`
struct pass
{
    /// A unique name used to identify the pass
    std::string name() const;
    /// Run the pass on the program
    void apply(program& p) const;
};

#else

<%
interface('pass',
    virtual('name', returns='std::string', const=True),
    virtual('apply', returns='void', p='program &', const=True)
)
%>

#endif

} // namespace migraph

#endif
