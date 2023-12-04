
#ifndef MIGRAPHX_GUARD_OPERATORS_RESIZE_HPP
#define MIGRAPHX_GUARD_OPERATORS_RESIZE_HPP

#include <array>
// #include <migraphx/op/common.hpp>
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

struct resize
{
    // TODO:   indicators.  The real scales and sizes are inputs, not attributes.
    std::vector<float> scales;
    std::vector<int64_t> sizes;
    int mode = 0; // 1: nereast 2: bilinear/linear 3: cubic
    std::string coordinate_transformation_mode;

    std::string name() const { return "resize"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.scales, "scales"),
                    f(self.sizes, "sizes"),
                    f(self.mode,"mode"),
                    f(self.coordinate_transformation_mode,"coordinate_transformation_mode"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // check_shapes{{inputs[0]}, *this, true}.has(2);
        check_shapes{inputs, *this, true}.has(2);

            // I get to DECIDE what the inputs are.  inputs are X, sizes or scale, ROI not supported

        if((sizes.empty()) == (scales.empty()))
            MIGRAPHX_THROW("RESIZE: One and only one of max_size or scales attributes must be given");
        if(inputs.back().ndim() != 1)
             MIGRAPHX_THROW("RESIZE: size/scale input must have rank 1");
        if(inputs.back().dynamic() and not inputs.back().dyn_dims()[0].is_fixed())
            MIGRAPHX_THROW("RESIZE: size/scale input must be fixed size");

        if(inputs.front().ndim() != inputs.back().to_static(1).lens()[0])
            MIGRAPHX_THROW("RESIZE: size/scale input's size must match rank of input X");
        if(not sizes.empty())
        {
            // the second shape is sizes
            
        }
        else
        {
            // the second shape is scales
            
        }
        // if(std::any_of(
        //     inputs.cbegin(), inputs.cend(), [](auto input) { return input->get_shape().dynamic(); }))
        // {
        // }

        // No matter what the inputs, the output shape is dynamic, with an unlimited size range.
        // TODO:  How can we tell if the input shape is a literal?  If it is, and input X is static, 
        // we can output a static shape.
        std::size_t max_val = std::numeric_limits<std::size_t>::max();
        std::vector<shape::dynamic_dimension> dyn_dims(inputs.back().lens().at(0),
                                                        shape::dynamic_dimension{0, max_val});
        return {inputs.front().type(), dyn_dims};


        // static input.  
        // if(!scales.empty())
        // {
        //     // 计算输出blob大小
        //     auto in_s    = inputs[0];
        //     auto in_lens = in_s.lens();
        //     if(in_lens.size() != scales.size())
        //     {
        //         MIGRAPHX_THROW("PARSE_UPSAMPLE: ranks of input and scale are different!");
        //     }
        //     std::vector<std::size_t> out_lens(in_lens.size());
        //     std::transform(in_lens.begin(),
        //                 in_lens.end(),
        //                 scales.begin(),
        //                 out_lens.begin(),
        //                 [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });
            
        //     return shape{in_s.type(), out_lens};
        // }
        // else if(!sizes.empty())
        // {
        //     return shape{inputs[0].type(), sizes};

        // }

        
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        // See scatter.hpp or gather.hpp for how to do a similar iteration with reduction
                // iterate through items in shape
        argument result{dyn_out.computed_shape};
        // negative axis means counting dimensions from back
        auto lens                 = args[0].get_shape().lens();
//Everything that follows is placeholder logic
        auto axis = 2;
        std::size_t axis_dim_size = lens[axis];
        // max dimension in axis
        visit_all(result, args[0])([&](auto output, auto data) {

            // the size input
            args[1].visit([&](auto indices) {
for(auto aa : indices ) std::cout << aa << "   indices \n";                
                if(dyn_out.computed_shape.scalar())
                {
 std::cout << " scalar output\n";
                }
                else
                {
                    // for each element in output, calculate index in input
                    for(auto bb : data) std::cout << bb << "   zzz data \n";

                    // auto out_lens  = data.get_shape().lens();
                    // out_lens[axis] = indices.get_shape().elements();
                    migraphx::shape out_comp_shape{data.get_shape().type(), indices};

                    shape_for_each(out_comp_shape, [&](const auto& out_idx_v, size_t out_idx) {
                        auto data_idx   = out_idx_v;
                        auto in_index   = indices[data_idx[axis]];
                        in_index        = (in_index < 0) ? in_index + axis_dim_size : in_index;
                        data_idx[axis]  = in_index;
                        output[out_idx] = data(data_idx.begin(), data_idx.end());
                        std::cout << " !!!!! did something\n";
                    });
                }
            });
        });
        return result;
    }

};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
