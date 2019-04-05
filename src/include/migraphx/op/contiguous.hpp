#ifndef MIGRAPHX_GUARD_OPERATORS_CONTIGUOUS_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONTIGUOUS_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/// The contiguous operator takes a non-standard input tensor and returns
/// the same tensor but in standard form. For example, if input tensor A which has lens = (4,5)
/// is first transposed, i.e. lens = (5,4), this tensor's data layout remained the same
/// during the transpose operation; only it's shape lengths and strides were changed.
/// This leaves the tensor in a non-standard form. The contiguous operator copies the
/// underlying data such that resulting tensor is returned to a standard form.
struct contiguous
{
    std::string name() const { return "contiguous"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto lens = inputs.at(0).lens();
        auto t    = inputs.at(0).type();
        return {t, lens};
    }
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        assert(output_shape.standard());
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            shape_for_each(output.get_shape(), [&](const auto& idx) {
                output(idx.begin(), idx.end()) = input(idx.begin(), idx.end());
            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
