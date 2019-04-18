#ifndef MIGRAPHX_GUARD_OPERATORS_CONVERT_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONVERT_HPP

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

struct convert
{
    shape::type_t targe_type = shape::half_type;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.targe_type, "target_type"));
    }

    std::string name() const { return "convert"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return {targe_type, inputs.front().lens(), inputs.front().strides()};
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
