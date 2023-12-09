
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

   // from parse_resize.cpp
auto& get_nearest_op(const std::string& near_mode)
{
    using nearest_op = std::function<std::size_t(std::size_t, double)>;
    static std::unordered_map<std::string, nearest_op> const nearest_ops = {
        {"round_prefer_floor",
        [=](std::size_t d_in, double val) {
            val = std::max(0.0, std::min(d_in - 1.0, val));
            return static_cast<std::size_t>(std::ceil((val - 0.5)));
        }},
        {"round_prefer_ceil",
        [=](std::size_t d_in, double val) {
            val = std::max(0.0, std::min(d_in - 1.0, val));
            return static_cast<std::size_t>(std::round((val)));
        }},
        {"floor",
        [=](std::size_t d_in, double val) {
            val = std::max(0.0, std::min(d_in - 1.0, val));
            return static_cast<std::size_t>(std::floor((val)));
        }},
        {"ceil", [=](std::size_t d_in, double val) {
            val = std::max(0.0, std::min(d_in - 1.0, val));
            return static_cast<std::size_t>(std::ceil((val)));
        }}};

    if(not contains(nearest_ops, near_mode))
    {
        MIGRAPHX_THROW("RESIZE: nearest_mode " + near_mode + " not supported!");
    }

    return nearest_ops.at(near_mode);
}

const auto& get_original_idx_op(const std::string& mode)
{
    using original_idx_op = std::function<double(std::size_t, std::size_t, std::size_t, double)>;
    static std::unordered_map<std::string, original_idx_op> const idx_ops = {
        {"half_pixel",
         [=](std::size_t, std::size_t, std::size_t idx, double scale) {
             return (idx + 0.5) / scale - 0.5;
         }},
        {"pytorch_half_pixel",
         [=](std::size_t, std::size_t l_out, std::size_t idx, double scale) {
             return l_out > 1 ? (idx + 0.5) / scale - 0.5 : 0.0;
         }},
        {"align_corners",
         [=](std::size_t l_in, std::size_t l_out, std::size_t idx, double) {
             return (l_out == 1) ? 0.0 : (1.0 * idx * (l_in - 1.0) / (l_out - 1.0));
         }},
        {"asymmetric",
         [=](std::size_t, std::size_t, std::size_t idx, double scale) { return idx / scale; }},
        {"tf_half_pixel_for_nn", [=](std::size_t, std::size_t, std::size_t idx, double scale) {
             return (idx + 0.5) / scale;
         }}};

    if(not contains(idx_ops, mode))
    {
        MIGRAPHX_THROW("RESIZE: coordinate_transformation_mode " + mode + " not supported!");
    }

    return idx_ops.at(mode);
}

struct resize
{
    // TODO:   indicators.  The real scales and sizes are inputs, not attributes.
    std::vector<float> scales;
    std::vector<int64_t> sizes;
    std::string nearest_mode;

    int mode = 0; // 1: nearest 2: bilinear/linear 3: cubic
    std::string coordinate_transformation_mode;

    std::string name() const { return "resize"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.scales, "scales"),
                    f(self.sizes, "sizes"),
                    f(self.nearest_mode,"nearest_mode"),
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

       
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        // See scatter.hpp or gather.hpp for how to do a similar iteration with reduction
        shape output_shape;
        auto in_lens = args[0].get_shape().to_static(1).lens();
        std::vector<size_t> out_lens(in_lens.size());        

        // Scales are either given, or calculated from output shape
        std::vector<double> vec_scale(in_lens.size());

        if(dyn_out.computed_shape.dynamic())
        {
            // calculate output shape from scales or sizes
            if(not sizes.empty())
            {
                // read sizes from args[1]  
                // out_lens = args[1].get_shape().to_static(1).lens();   //     <===
                // Compute the scales from the given output dimensions

                // Copy the output size
                args[1].visit([&](auto size_input) {
                    for(auto aa : size_input ) std::cout << aa << "   sizes \n";

                    std::transform(size_input.begin(), size_input.end(), out_lens.begin(),
                                [](auto size_i) { 
std::cout << size_i  << "   transform \n";                                    
                            return size_i;                     
                    }); 
std::cout << "***\n";
                    for(auto aa : out_lens ) std::cout << aa << "   out_lens \n";  
std::cout << "***\n";

                // Deduce the scales for each axis
                std::transform(size_input.begin(), size_input.end(), in_lens.begin(), vec_scale.begin(),
                                [](auto sz, size_t in_len) {
                            return static_cast<double>(sz)/in_len;                     
                    });                   
                });          

                for(auto aa : vec_scale ) std::cout << aa << "   vec_scale \n";                
            }
            else
            {
                args[1].visit([&](auto scale_input) {
                    for(auto aa : scale_input ) std::cout << aa << "   scale_input \n";                
                    // read the scale from args[1]-- vec_scale = scale_input;
                    //
                    std::transform(scale_input.begin(), scale_input.end(), vec_scale.begin(),
                                [](auto scale_i) {
                            return scale_i;                     
                    }); 

                    // compute the output dimensions from the given scale
                    std::transform(scale_input.begin(), scale_input.end(), in_lens.begin(), out_lens.begin(),
                                [](auto scale_i, size_t in_len) {
                            return static_cast<size_t>(scale_i*in_len);                     
                    });                    
                });
            }
            output_shape = {args[0].get_shape().type(), out_lens};
        }

        argument result{output_shape};
        auto nearest_op = get_nearest_op(nearest_mode);
        auto idx_op              = get_original_idx_op(coordinate_transformation_mode);

        // temp.  This is a placeholder for reading the desired dimensions or scale

       // max dimension in axis
        visit_all(result, args[0])([&](auto output, auto data) {

            // the size input
            // args[1].visit([&](auto indices) {
            //     for(auto aa : indices ) std::cout << aa << "   indices \n";                
            //     if(dyn_out.computed_shape.scalar())
            //     {
            //         std::cout << " scalar output\n";
            //     }
            //     else
            //     {
                    // for each element in output, calculate index in input
                    for(auto bb : data) std::cout << bb << "   zzz data \n";

                    migraphx::shape out_comp_shape{data.get_shape().type(), out_lens};

                    shape_for_each(out_comp_shape, [&](const auto& out_idx_v, size_t out_idx) {
                        std::vector<size_t> in_idx(out_idx_v.size());
                        for(auto ii = 0; ii < out_idx_v.size(); ++ii)
                        {
                            auto idx_val = idx_op(in_lens[ii], out_lens[ii], out_idx_v[ii], vec_scale[ii]);
                            in_idx[ii]   = nearest_op(in_lens[ii], idx_val);
                        }                  
                        std::cout << "\n";
                        std::cout <<out_idx <<  "  out_index\n"; 
auto zap = data(in_idx.begin(), in_idx.end());

for(auto gg : output)  std::cout << gg <<  "  "; std::cout <<"ggg\n";
                        // use index function instead?
                        output[out_idx] = data(in_idx.begin(), in_idx.end());
                        std::cout << zap <<  "\n";
                    });
            //     }
            // });
        });
        std::cout << " finish resize\n";
        return result;
    }

};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
