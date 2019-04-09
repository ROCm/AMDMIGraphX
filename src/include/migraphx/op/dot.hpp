#ifndef MIGRAPHX_GUARD_OPERATORS_DOT_HPP
#define MIGRAPHX_GUARD_OPERATORS_DOT_HPP

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

struct dot
{
    float alpha = 1.0;
    float beta  = 1.0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"), f(self.beta, "beta"));
    }

    std::string name() const { return "dot"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.same_type();
        const shape& a = inputs.at(0);
        const shape& b = inputs.at(1);
        auto t         = a.type();

        if(!std::all_of(inputs.begin(), inputs.end(), [](auto s) { return s.lens().size() >= 2; }))
        {
            MIGRAPHX_THROW("DOT: dot only accept 2 or more dims operands");
        }

        // only handle the case that the batch size of a and b are the same
        if(!std::equal(
               a.lens().rbegin() + 2, a.lens().rend(), b.lens().rbegin() + 2, b.lens().rend()))
        {
            MIGRAPHX_THROW("DOT: batch size of A and B mismatch: {" + to_string_range(a.lens()) +
                           "} x {" + to_string_range(b.lens()) + "}");
        }

        std::size_t dim_0 = a.lens().size() - 2;
        std::size_t dim_1 = a.lens().size() - 1;
        if(a.lens()[dim_1] != b.lens()[dim_0])
        {
            MIGRAPHX_THROW("DOT: inner dimensions do not match: {" + to_string_range(a.lens()) +
                           "} x {" + to_string_range(b.lens()) + "}");
        }

        auto out_lens   = a.lens();
        out_lens[dim_1] = b.lens()[dim_1];
        if(inputs.size() == 3 && out_lens != inputs.at(2).lens())
        {
            MIGRAPHX_THROW("DOT: dimension mismatch, operand C: {" +
                           to_string_range(inputs.at(2).lens()) +
                           "}, cannot add to operand A * B: {" + to_string_range(out_lens) + "}");
        }

        return {t, out_lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
