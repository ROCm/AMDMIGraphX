#ifndef MIGRAPHX_GUARD_PASS_HPP
#define MIGRAPHX_GUARD_PASS_HPP

#include <cassert>
#include <string>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/rank.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;
struct module;
struct module_pass_manager;

#ifdef DOXYGEN

/// An interface for applying a transformation to the instructions in a
/// `program`
struct pass
{
    /// A unique name used to identify the pass
    std::string name() const;
    /// Run the pass on the module
    void apply(module_pass_manager& mpm) const;
    void apply(module& m) const;
    /// Run the pass on the program
    void apply(program& p) const;
};

#else

module& get_module(module_pass_manager& mpm);

namespace detail {

template <class T>
auto module_pass_manager_apply(rank<1>, const T& x, module_pass_manager& mpm)
    -> decltype(x.apply(get_module(mpm)))
{
    return x.apply(get_module(mpm));
}

template <class T>
void module_pass_manager_apply(rank<0>, const T&, module_pass_manager&)
{
}

template <class T>
void module_pass_manager_apply(const T& x, module_pass_manager& mpm)
{
    module_pass_manager_apply(rank<1>{}, x, mpm);
}

} // namespace detail

<%
interface('pass',
    virtual('name', returns='std::string', const=True),
    virtual('apply', returns='void', mpm='module_pass_manager &', const=True, default='migraphx::detail::module_pass_manager_apply'),
    virtual('apply', returns='void', p='program &', const=True, default='migraphx::nop')
)
%>

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
