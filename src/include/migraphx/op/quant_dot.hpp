#ifndef MIGRAPHX_GUARD_OPERATORS_QUANT_DOT_HPP
#define MIGRAPHX_GUARD_OPERATORS_QUANT_DOT_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct quant_dot
{
    value attributes() const { return {{"general_data_type", "dot"}}; }

    std::string name() const { return "quant_dot"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{{inputs.at(0), inputs.at(1)}, *this}.same_type().has(2);
        const shape& a = inputs.at(0);
        const shape& b = inputs.at(1);
        auto t         = a.type();
        if(t != shape::int8_type)
        {
            MIGRAPHX_THROW("QUANT_DOT: only support data type int8_t");
        }

        if(!std::all_of(inputs.begin(), inputs.end(), [](auto s) { return s.lens().size() >= 2; }))
        {
            MIGRAPHX_THROW("QUANT_DOT: dot only accept 2 or more dims operands");
        }

        // only handle the case that the batch size of a and b are the same
        if(!std::equal(
               a.lens().rbegin() + 2, a.lens().rend(), b.lens().rbegin() + 2, b.lens().rend()))
        {
            MIGRAPHX_THROW("QUANT_DOT: batch size of A and B mismatch: {" +
                           to_string_range(a.lens()) + "} x {" + to_string_range(b.lens()) + "}");
        }

        std::size_t dim_0 = a.lens().size() - 2;
        std::size_t dim_1 = a.lens().size() - 1;
        if(a.lens()[dim_1] != b.lens()[dim_0])
        {
            MIGRAPHX_THROW("QUANT_DOT: inner dimensions do not match: {" +
                           to_string_range(a.lens()) + "} x {" + to_string_range(b.lens()) + "}");
        }

        auto out_lens   = a.lens();
        out_lens[dim_1] = b.lens()[dim_1];
        return {shape::int32_type, out_lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
