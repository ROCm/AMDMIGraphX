#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_OPERAND_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_OPERAND_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/reflect.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/auto_any_cast.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct context;

#ifdef DOXYGEN

/// The operation interface represents an action an instruction will perform. All
/// operation classes must be CopyConstructible.
struct operation
{
    /// A unique name identifying the operation
    std::string name() const;
    /// An optional method that can be used to finalize the operator before running
    void finalize(context& ctx);
    /// This is used to compute the resulting shape from an operation. If an
    /// operation cannot be run with input shapes, then it should throw an
    /// exception.
    shape compute_shape(const std::vector<shape>& input) const;
    /**
     * @brief This performs the operation's computation.
     *
     * This method can be optional when the operation is only used as a placeholder to be lowered
     * later on.
     *
     * @param ctx This is the context created by the `target` during compilation. Implementations
     * can use the target's `context` class rather than the `context` interface class.
     * @param output This is the output shape. It is equivalent to running `compute_shape` with each
     * `shape` of the `argument`.
     * @param input This is the `argument` result from the previous instruction's computation.
     * @return Return an `argument` of the result computation. The `shape` of `argument` should be
     * the same the `output` shape.
     */
    argument compute(context& ctx, const shape& output, const std::vector<argument>& input) const;
    /// An optional method to return which argument the output will alias. If
    /// there is no aliased output then -1 can be returned.
    std::ptrdiff_t output_alias(const std::vector<shape>& input) const;
    /// An optional stream operator to print the operation. When this is not
    /// implemented, it will just print the operation's name.
    friend std::ostream& operator<<(std::ostream& os, const operation& op);
};

/// Returns true if operation does not require a context to run compute
bool is_context_free(const operation& x);
/// Returns true if the operation has a finalize method
bool has_finalize(const operation& x);

#else

namespace detail {

namespace operation_operators {

template <class T>
auto operator<<(std::ostream& os, const T& x) -> decltype(os << x.name())
{
    os << x.name();
    char delim = '[';
    reflect_each(x, [&](auto&& y, auto name) {
        os << delim;
        os << name << "=";
        stream_write_value(os, y);
        delim = ',';
    });
    if(delim == ',')
        os << "]";
    return os;
}

template <class T, class U>
auto operator==(const T& x, const U& y) -> decltype(x.name() == y.name())
{
    static_assert(is_reflectable<T>{} or sizeof(T) <= 1,
                  "Missing equality operator or reflect method.");
    if(x.name() != y.name())
        return false;
    const auto& yy = any_cast<T>(y);
    return reflect_tie(x) == reflect_tie(yy);
}

} // namespace operation_operators

template <class T>
auto compute_op(rank<2>,
                const T& x,
                context& ctx,
                const shape& output_shape,
                const std::vector<argument>& input)
    -> decltype(x.compute(auto_any_cast(ctx), output_shape, input))
{
    return x.compute(auto_any_cast(ctx), output_shape, input);
}

template <class T>
auto compute_op(
    rank<1>, const T& x, context&, const shape& output_shape, const std::vector<argument>& input)
    -> decltype(x.compute(output_shape, input))
{
    return x.compute(output_shape, input);
}

template <class T>
argument compute_op(rank<0>, const T& x, context&, const shape&, const std::vector<argument>&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T>
argument
compute_op(const T& x, context& ctx, const shape& output_shape, const std::vector<argument>& input)
{
    return compute_op(rank<2>{}, x, ctx, output_shape, input);
}

template <class T>
auto compute_op(rank<2>, const T& x, const shape& output_shape, const std::vector<argument>& input)
    -> decltype(x.compute(output_shape, input))
{
    return x.compute(output_shape, input);
}

template <class T>
auto compute_op(rank<1>, const T& x, const shape& output_shape, const std::vector<argument>& input)
    -> decltype(x.compute(auto_any_cast(std::declval<context&>()), output_shape, input))
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable without a context: " + name);
}

template <class T>
argument compute_op(rank<0>, const T& x, const shape&, const std::vector<argument>&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T>
argument compute_op(const T& x, const shape& output_shape, const std::vector<argument>& input)
{
    return compute_op(rank<2>{}, x, output_shape, input);
}

template <class T>
auto is_context_free_op(rank<1>,
                        const T& x,
                        const shape& output_shape,
                        const std::vector<argument>& input)
    -> decltype(x.compute(output_shape, input), std::true_type{});

template <class T>
auto is_context_free_op(rank<0>, const T&, const shape&, const std::vector<argument>&)
    -> std::false_type;

template <class T>
auto is_context_free_op(const T& x) -> decltype(is_context_free_op(
    rank<1>{}, x, std::declval<const shape&>(), std::declval<std::vector<argument>>()))
{
    return {};
}

template <class T>
std::ptrdiff_t output_alias_op(const T&, const std::vector<shape>&)
{
    return -1;
}

template <class T>
auto finalize_op(
    rank<1>, T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
    -> decltype(x.finalize(auto_any_cast(ctx), output_shape, input), void())
{
    x.finalize(auto_any_cast(ctx), output_shape, input);
}

template <class T>
void finalize_op(rank<0>, T&, context&, const shape&, const std::vector<shape>&)
{
}

template <class T>
void finalize_op(T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
{
    finalize_op(rank<1>{}, x, ctx, output_shape, input);
}

template <class T>
auto has_finalize_op(
    rank<1>, T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
    -> decltype(x.finalize(auto_any_cast(ctx), output_shape, input), std::true_type{});

template <class T>
auto has_finalize_op(rank<0>, T&, context&, const shape&, const std::vector<shape>&)
    -> std::false_type;

template <class T>
auto has_finalize_op(const T&) -> decltype(has_finalize_op(rank<1>{},
                                                           std::declval<T&>(),
                                                           std::declval<context&>(),
                                                           std::declval<const shape&>(),
                                                           std::declval<std::vector<shape>>()))
{
    return {};
}

} // namespace detail

<%
 interface(
     'operation',
     virtual('name', returns = 'std::string', const = True),
     virtual('is_context_free', returns = 'bool', const = True, default = 'detail::is_context_free_op'),
     virtual('has_finalize', returns = 'bool', const = True, default = 'detail::has_finalize_op'),
     virtual('output_alias',
             returns = 'std::ptrdiff_t',
             input   = 'const std::vector<shape>&',
             const   = True,
             default = 'detail::output_alias_op'),
     virtual('finalize',
             ctx     = 'context&',
             output  = 'const shape&',
             input   = 'const std::vector<shape>&',
             default = 'detail::finalize_op'),
     virtual('compute_shape', returns = 'shape', input = 'const std::vector<shape>&', const = True),
     virtual('compute',
             returns = 'argument',
             ctx     = 'context&',
             output  = 'const shape&',
             input   = 'const std::vector<argument>&',
             const   = True,
             default = 'detail::compute_op'),
     virtual('compute',
             returns = 'argument',
             output  = 'const shape&',
             input   = 'const std::vector<argument>&',
             const   = True,
             default = 'detail::compute_op'),
     friend('operator<<',
            returns = 'std::ostream &',
            os      = 'std::ostream &',
            op      = 'const operation &',
            using   = 'migraphx::detail::operation_operators::operator<<'),
     friend('operator==',
            returns = 'bool',
            x       = 'const operation &',
            y       = 'const operation &',
            using   = 'migraphx::detail::operation_operators::operator==')) %>

    inline bool operator!=(const operation& x, const operation& y)
{
    return !(x == y);
}

inline bool is_context_free(const operation& op) { return op.is_context_free(); }

template <class T>
bool is_context_free(const T& x)
{
    return detail::is_context_free_op(x);
}

inline bool has_finalize(const operation& op) { return op.has_finalize(); }

template <class T>
bool has_finalize(const T& x)
{
    return detail::has_finalize_op(x);
}

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
