#ifndef MIGRAPHX_GUARD_OPERATORS_QUANT_DOT_HPP
#define MIGRAPHX_GUARD_OPERATORS_QUANT_DOT_HPP

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

struct quant_dot
{
    int32_t alpha = 1;
    int32_t beta  = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(as_number(self.alpha), "alpha"), f(as_number(self.beta), "beta"));
    }

    std::string name() const { return "quant_dot"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{{inputs.at(0), inputs.at(1)}, *this}.same_type();
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

        // k be multiple of 4
        if((a.lens()[dim_1] % 4) != 0)
        {
            MIGRAPHX_THROW("QUANT_DOT: size of A {" + to_string_range(a.lens()) + "} and B {" +
                           to_string_range(b.lens()) + "} must be multiple of 4 for int8 type");
        }

        auto out_lens   = a.lens();
        out_lens[dim_1] = b.lens()[dim_1];
        if(inputs.size() == 3 && out_lens != inputs.at(2).lens())
        {
            MIGRAPHX_THROW("QUANT_DOT: dimension mismatch, operand C: {" +
                           to_string_range(inputs.at(2).lens()) +
                           "}, cannot add to operand A * B: {" + to_string_range(out_lens) + "}");
        }

        if(inputs.size() == 3 && inputs.at(2).type() != shape::int32_type)
        {
            MIGRAPHX_THROW("QUANT_DOT: operand C type must be int32");
        }

        return {shape::int32_type, out_lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
