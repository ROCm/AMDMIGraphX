#ifndef MIGRAPH_GUARD_MIGRAPHLIB_OPERAND_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_OPERAND_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraph/shape.hpp>
#include <migraph/argument.hpp>
#include <migraph/context.hpp>
#include <migraph/auto_any_cast.hpp>

namespace migraph {

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
    shape compute_shape(std::vector<shape> input) const;
    /**
     * @brief This performs the operation's computation
     *
     * @param ctx This is the context created by the `target` during compilation. Implementations
     * can use the target's `context` class rather than the `context` interface class.
     * @param output This is the output shape. It is equivalent to running `compute_shape` with each
     * `shape` of the `argument`.
     * @param input This is the `argument` result from the previous instuction's computation.
     * @return Return an `argument` of the result computation. The `shape` of `argument` should be
     * the same the `output` shape.
     */
    argument compute(context& ctx, shape output, std::vector<argument> input) const;
    /// An optional stream operator to print the operation. When this is not
    /// implemented, it will just print the operation's name.
    friend std::ostream& operator<<(std::ostream& os, const operation& op);
};

#else

namespace operation_stream {

template <class T>
auto operator<<(std::ostream& os, const T& x) -> decltype(os << x.name())
{
    return os << x.name();
}

} // namespace operation_stream

template <class T>
argument compute_op(const T& x, context& ctx, shape output_shape, std::vector<argument> input)
{
    return x.compute(auto_any_cast(ctx), output_shape, input);
}

<%
interface('operation',
    virtual('name', returns='std::string', const=True),
    virtual('compute_shape', returns='shape', input='std::vector<shape>', const=True),
    virtual('compute', returns='argument', ctx='context&', output='shape', input='std::vector<argument>', const=True, default='compute_op'),
    friend('operator<<', returns='std::ostream &', os='std::ostream &', op='const operation &', using='migraph::operation_stream::operator<<')
)
%>

#endif

} // namespace migraph

#endif
