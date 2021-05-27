#ifndef MIGRAPHX_GUARD_OPERATORS_REVERSE_HPP
#define MIGRAPHX_GUARD_OPERATORS_REVERSE_HPP

#include <algorithm>
#include <vector>
#include <cmath>
#include <utility>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/argument.hpp>


namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reverse
{

    std::vector<int64_t> axis; //1-D, which axis will be reversed.

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "reverse"; }

    value attributes() const
    {
         value normalize;
         normalize["axis"] = value::array{normalize_attribute::include_min};
         return {{"normalize_axes", normalize}};
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        auto lens = inputs[0].lens();
        auto type = inputs[0].type();
        return shape{type,lens};
    }

    // shape compute_shape(std::vector<shape> inputs) const
    // {
    //     return shape{inputs.front().type(), inputs.front().lens()};
    // }

    argument compute(const shape&, std::vector<argument> args) const
    {
        auto input  = args[0].get_shape(); //float_type, {2, 16}, {16, 1}

        std::vector<std::size_t> data; 
        args[0].visit([&](auto s) { data.assign(s.begin(), s.end()); }); 

        const std::vector<std::size_t>& lens = input.lens();
        //const std::vector<std::size_t>& strides = input.strides();

        if (axis[0] == 0)
        {
            for(std::size_t i = 0; i < (lens[0] / 2); i++)
            {
                //TODO
                continue;
            }
        }
        else if (axis[0] == 1) //outer to inside, this is the sub arrays
        {
            for (std::size_t t = 0; t < lens[0]; t++) //4096
            {
                std::reverse( data.begin() + (t * lens[1]), data.begin() + ((t+1) * lens[1]) ); //
            }
        }
        else {
            MIGRAPHX_THROW("reverse op can only have axis=1 or axis=0");
        }

        argument result{input};
        
        result.visit([&](auto output) {
            par_for(input.elements(), [&](auto i) {
                output[i]     = data[i];
            });
        });

        return result;
    }
    
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
