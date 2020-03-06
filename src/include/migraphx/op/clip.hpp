#ifndef MIGRAPHX_GUARD_OPERATORS_CLIP_HPP
#define MIGRAPHX_GUARD_OPERATORS_CLIP_HPP

#include <array>
#include <migraphx/op/unary.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>
#include <limits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct clip
{
    std::string name() const { return "clip"; } 

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(3).same_type();
        return inputs.front();
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        
        visit_all(result, args[0], args[1], args[2])([&](auto output, auto input, auto min_val, auto max_val) {
            auto max = max_val.front();
            auto min = min_val.front();
            std::transform(input.begin(),
                            input.end(),
                            output.begin(),
                            [max, min](auto x){
                                using type = decltype(x);
                                return std::min(std::max(type(min), x), type(max));
                            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
