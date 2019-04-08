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
    float beta  = 0.0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"), f(self.beta, "beta"));
    }

    std::string name() const { return "dot"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type();
        const shape& a = inputs.at(0);
        const shape& b = inputs.at(1);
        auto t         = a.type();

        // according to the specification of the numpy.matmul()
        // inputs with the shape dims more than 2 are acceptable
        // as long as dim values are the same in the two inputs
        if(!std::equal(a.lens().rbegin() + 2, a.lens().rend(), b.lens().rbegin() + 2))
        {
            MIGRAPHX_THROW("DOT: dim values mismatch");
        }

        std::size_t dim_0 = a.lens().size() - 2;
        std::size_t dim_1 = a.lens().size() - 1;
        if(a.lens()[dim_1] != b.lens()[dim_0])
            MIGRAPHX_THROW("Inner dimensions do not match: {" + to_string_range(a.lens()) +
                           "} x {" + to_string_range(b.lens()) + "}");
        auto out_lens   = a.lens();
        out_lens[dim_1] = b.lens()[dim_1];
        return {t, out_lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
