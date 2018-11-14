#ifndef MIGRAPH_GUARD_MIGRAPHLIB_OPERAND_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_OPERAND_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/shape.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/context.hpp>
#include <migraphx/auto_any_cast.hpp>

namespace migraphx {

#ifdef DOXYGEN

/// The operation interface represents an action an instruction will perform. All
/// operation classes must be CopyConstructible.
struct operation
{
    /// A unique name identifying the operation
    std::string name() const;
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
    int output_alias(const std::vector<shape>& input) const;
    /// An optional stream operator to print the operation. When this is not
    /// implemented, it will just print the operation's name.
    friend std::ostream& operator<<(std::ostream& os, const operation& op);
};

#else

namespace operation_stream {

template <class T>
auto operator<<(std::ostream& os, const T& x) -> decltype(os << x.name())
{
    os << x.name();
    char delim = '[';
    reflect_each(x, [&](auto& y, auto name) {
        os << delim;
        os << name << "=";
        stream_write_value(os, y);
        delim = ',';
    });
    if(delim == ',')
        os << "]";
    return os;
}

} // namespace operation_stream

namespace operation_equal {

template <class T, class U>
auto operator==(const T& x, const U& y) -> decltype(x.name() == y.name())
{
    if(x.name() != y.name())
        return false;
    const auto& yy = any_cast<T>(y);
    return reflect_tie(x) == reflect_tie(yy);
}

} // namespace operation_equal

template <class T>
auto compute_op(rank<1>,
                const T& x,
                context& ctx,
                const shape& output_shape,
                const std::vector<argument>& input)
    -> decltype(x.compute(auto_any_cast(ctx), output_shape, input))
{
    return x.compute(auto_any_cast(ctx), output_shape, input);
}

template <class T>
argument compute_op(rank<0>, const T& x, context&, const shape&, const std::vector<argument>&)
{
    std::string name = x.name();
    MIGRAPH_THROW("Not computable: " + name);
}

template <class T>
argument
compute_op(const T& x, context& ctx, const shape& output_shape, const std::vector<argument>& input)
{
    return compute_op(rank<1>{}, x, ctx, output_shape, input);
}

template <class T>
int output_alias_op(rank<0>, const T&, const std::vector<shape>&)
{
    return -1;
}

template <class T>
auto output_alias_op(rank<1>, const T& x, const std::vector<shape>& shapes)
    -> decltype(x.output_alias(shapes))
{
    return x.output_alias(shapes);
}

template <class T>
int output_alias_op(const T& x, const std::vector<shape>& shapes)
{
    return output_alias_op(rank<1>{}, x, shapes);
}

<%
 interface(
     'operation',
     virtual('name', returns = 'std::string', const = True),
     virtual('output_alias',
             returns = 'int',
             input   = 'const std::vector<shape>&',
             const   = True,
             default = 'output_alias_op'),
     virtual('compute_shape', returns = 'shape', input = 'const std::vector<shape>&', const = True),
     virtual('compute',
             returns = 'argument',
             ctx     = 'context&',
             output  = 'const shape&',
             input   = 'const std::vector<argument>&',
             const   = True,
             default = 'compute_op'),
     friend('operator<<',
            returns = 'std::ostream &',
            os      = 'std::ostream &',
            op      = 'const operation &',
            using   = 'migraphx::operation_stream::operator<<'),
     friend('operator==',
            returns = 'bool',
            x       = 'const operation &',
            y       = 'const operation &',
            using   = 'migraphx::operation_equal::operator==')) %>

    inline bool operator!=(const operation& x, const operation& y)
{
    return !(x == y);
}

#endif

} // namespace migraphx

#endif
