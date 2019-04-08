#ifndef MIGRAPHX_GUARD_OPERATORS_FP_CONVERSION_HPP
#define MIGRAPHX_GUARD_OPERATORS_FP_CONVERSION_HPP

#include <array>
#include <migraphx/op/binary.hpp>
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

struct fp_conversion
{
    bool reduce_precision = true;
    std::string name() const { return "fp_conversion"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        if(reduce_precision)
        {
            if(inputs.front().type() != shape::float_type)
            {
                MIGRAPHX_THROW("FP_CONVERSION: input arguments must be type float");
            }

            return {shape::half_type, inputs.front().lens()};
        }
        else
        {
            if(inputs.front().type() != shape::half_type)
            {
                MIGRAPHX_THROW("FP_CONVERSION: input arguments must be type fp16");
            }

            return {shape::float_type, inputs.front().lens()};
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        result.visit([&](auto output) {
            args.front().visit(
                [&](auto input) { std::copy(input.begin(), input.end(), output.begin()); });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
