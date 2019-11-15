#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_TARGET_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_TARGET_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <migraphx/context.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/config.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/rank.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// An interface for a compilation target
struct target
{
    /// A unique name used to identify the target
    std::string name() const;
    /**
     * @brief The transformation pass to be run during compilation.
     *
     * @param ctx This is the target-dependent context that is created by `get_context`
     * @param options Compiling options passed in by the user
     * @return The passes to be ran
     */
    std::vector<pass> get_passes(context& ctx, const compile_options& options) const;
    /**
     * @brief Construct a context for the target.
     * @return The context to be used during compilation and execution.
     */
    context get_context() const;
    /**
     * @brief copy an argument to the current target.
     *
     * @param arg Input argument to be copied to the target
     * @return Argument in the target.
     */
    argument copy_to(const argument& arg) const;
    /**
     * @brief copy an argument from the current target.
     *
     * @param arg Input argument to be copied from the target
     * @return Argument in the host.
     */
    argument copy_from(const argument& arg) const;
    /**
     * @brief Allocate an argument based on the input shape
     *
     * @param s Shape of the argument to be allocated in the target
     * @return Allocated argument in the target.
     */
    argument allocate(const shape& s) const;
};

#else

template <class T>
auto target_allocate(rank<1>, T& x, const shape& s) -> decltype(x.allocate(s))
{
    return x.allocate(s);
}

template <class T>
argument target_allocate(rank<0>, T& x, const shape&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T>
argument target_allocate(T& x, const shape& s)
{
    return target_allocate(rank<1>{}, x, s);
}

template <class T>
auto copy_to_target(rank<1>, T& x, const argument& arg) -> decltype(x.copy_to(arg))
{
    return x.copy_to(arg);
}

template <class T>
argument copy_to_target(rank<0>, T&, const argument& arg)
{
    return arg;
}

template <class T>
argument copy_to_target(T& x, const argument& arg)
{
    return copy_to_target(rank<1>{}, x, arg);
}

template <class T>
auto copy_from_target(rank<1>, T& x, const argument& arg) -> decltype(x.copy_from(arg))
{
    return x.copy_from(arg);
}

template <class T>
argument copy_from_target(rank<0>, T&, const argument& arg)
{
    return arg;
}

template <class T>
argument copy_from_target(T& x, const argument& arg)
{
    return copy_from_target(rank<1>{}, x, arg);
}

<%
interface('target',
     virtual('name', returns='std::string', const=True),
     virtual('get_passes', ctx='context&', options='const compile_options&', returns='std::vector<pass>', const=True),
     virtual('get_context', returns='context', const=True),
     virtual('copy_to',
             returns = 'argument',
             input   = 'const argument&',
             const   = True,
             default = 'copy_to_target'),
     virtual('copy_from',
             returns = 'argument',
             input   = 'const argument&',
             const   = True,
             default = 'copy_from_target'),
    virtual('allocate', s='const shape&', returns='argument', const=True,
             default = 'target_allocate')
)
%>

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
