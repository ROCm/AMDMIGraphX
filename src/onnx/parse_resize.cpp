/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_resize : op_parser<parse_resize>
{
    std::vector<op_desc> operators() const
    {
        return {{"Resize", "resize"}, {"Upsample", "upsample"}};
    }

    struct resize_attr
    {
        std::vector<int64_t> axes;                // resize - 18
        int antialias             = 0;            // resize - 18
        int exclude_outside       = 0;            // resize - 11
        float cubic_coeff_a       = -0.75f;       // resize - 11
        float extrapolation_value = 0.0f;         // resize - 11
        std::string coord_t_mode  = "half_pixel"; // resize - 11
        std::string nearest_mode  = "round_prefer_floor";
        std::string keep_aspect   = "stretch"; // resize - 18

        // Overlaps with upsample operator
        std::string mode = "nearest";

        // Upsample related
        std::vector<float> scales = {}; // Upsample 7
    };

    struct resize_args
    {
        // Since inception opset(10)
        instruction_ref x;

        // For Upscale this may be an attr
        std::optional<instruction_ref> scales; // resize/upsample-10

        // Added in resize - 11
        // resize 13 makes roi optional
        std::optional<instruction_ref> roi;
        std::optional<instruction_ref> sizes;

        resize_attr r_attr;

        shape in_s;
        std::vector<size_t> in_lens;
        std::vector<size_t> out_lens;
        std::vector<float> vec_scale;
        instruction_ref scales_sizes_arg;

        int opset_version = -1;

        // if scale an attr must be greater or equal to 1
        bool is_scale_attr() const { return not r_attr.scales.empty(); }

        bool is_axes_used() const { return not r_attr.axes.empty(); }

        bool is_constant_scale_input() const { return not vec_scale.empty(); }

        std::string get_nearest_mode() const { return r_attr.nearest_mode; }
        std::string get_coord_trans_mode() const { return r_attr.coord_t_mode; }
        std::string get_mode() const { return r_attr.mode; }

        void set_scales_sizes_arg(instruction_ref ref) { scales_sizes_arg = ref; }

        instruction_ref get_scales_sizes_arg() const { return scales_sizes_arg; }

        void check_scales_and_inputs() const
        {
            if(in_lens.size() != vec_scale.size())
            {
                MIGRAPHX_THROW("PARSE_RESIZE: ranks of input and scale are different!");
            }
        }

        bool is_output_not_set() const
        {
            return all_of(out_lens.cbegin(), out_lens.cend(), [](auto o) { return o == 0; });
        }

        void compute_output_sizes()
        {
            std::transform(
                in_lens.begin(),
                in_lens.end(),
                vec_scale.begin(),
                out_lens.begin(),
                [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });
        }

        void compute_scales()
        {
            vec_scale.resize(in_lens.size());
            std::transform(in_lens.begin(),
                           in_lens.end(),
                           out_lens.begin(),
                           vec_scale.begin(),
                           [](auto iss, auto oss) { return 1.0 * oss / iss; });
        }

        void assign_scale_or_size(const std::vector<instruction_ref>& args)
        {
            set_scales_sizes_arg(args[0]);
            if(not is_constant_scale_input())
            {
                // Depending on the args, it *must* populate the `vec_scale`, and might populate
                // `out_lens`. Skip first input and `roi` input (if present)
                size_t args_offset = args.size() > 2 ? 2 : 1;
                std::vector<instruction_ref> inputs{args.begin() + args_offset, args.end()};
                for(const auto& arg : inputs)
                {
                    if(is_arg_invalid(arg))
                        continue;

                    scales_sizes_arg = arg;
                    auto arg_out     = arg->eval();

                    auto type = arg->get_shape().type();
                    if(is_arg_skipped(arg_out))
                        break;

                    if(type == shape::int64_type)
                    { // When input is using sizes
                        assign_output_sizes(arg_out);
                        check_output_size();
                        compute_scales();
                        break;
                    }
                    else if(type == shape::float_type)
                    { // When input is using scales
                        if(is_scale_rank_valid(arg))
                        {
                            assign_scales(arg_out);
                        }
                        break;
                    }
                    else
                    {
                        MIGRAPHX_THROW("PARSE_RESIZE: invalid shape type ");
                    }
                }

                if(vec_scale.empty() and out_lens.empty())
                    MIGRAPHX_THROW("PARSE_RESIZE: no shapes for scales/size input provided");
            }

            if(is_constant_scale_input())
            {
                check_scales_and_inputs();

                if(is_output_not_set())
                {
                    compute_output_sizes();
                }
            }
        }

        void set_coord_trans_mode(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "coordinate_transformation_mode"))
            {
                auto coord_trans_mode = attr.at("coordinate_transformation_mode").s();
                // does not support transformation mode "tf_crop_and_resize"
                if(coord_trans_mode == "tf_crop_and_resize")
                {
                    MIGRAPHX_THROW("PARSE_RESIZE: \"tf_crop_and_resize\" mode is not supported!");
                }
                r_attr.coord_t_mode = coord_trans_mode;
            }
        }

        void set_cubic_coeff(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "cubic_coeff_a"))
            {
                auto coeff           = attr.at("cubic_coeff_a").f();
                r_attr.cubic_coeff_a = coeff;
            }
        }

        void set_mode(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "mode"))
            { // TODO: Add support for cubic mode
                auto mode = attr.at("mode").s();
                if(mode != "nearest" and mode != "linear")
                {
                    MIGRAPHX_THROW("PARSE_RESIZE: only nearest and linear modes are supported!");
                }
                r_attr.mode = mode;
            }
        }

        void set_nearest_mode(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "nearest_mode"))
            {
                r_attr.nearest_mode = attr.at("nearest_mode").s();
            }
        }

        void set_exclude_outside(const onnx_parser::attribute_map& attr)
        {
            // TODO: Add support for exclude outside = 1
            if(contains(attr, "exclude_outside") and attr.at("exclude_outside").i() == 1)
            {
                MIGRAPHX_THROW("PARSE_RESIZE exclude_outside 1 is not supported!");
            }
        }

        void set_extrapolation_val(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "extrapolation_value"))
            {
                r_attr.extrapolation_value = attr.at("extrapolation_value").f();
            }
        }

        void set_axes(const onnx_parser::attribute_map& attr)
        {
            // TODO: support implementation of 'axes' attribute.
            // For now, it's used to check the length of 'sizes' input (if present)
            if(contains(attr, "axes"))
            {
                auto&& axes_vals = attr.at("axes").ints();
                r_attr.axes      = std::vector<int64_t>(axes_vals.begin(), axes_vals.end());
            }
        }

        void set_aspect_ratio_policy(const onnx_parser::attribute_map& attr,
                                     const std::vector<instruction_ref>& args) const
        {
            // TODO: Add support for this instead of keeping it as a check
            if(contains(attr, "keep_aspect_ratio_policy"))
            {
                shape last_arg_shape     = args.back()->get_shape();
                size_t last_arg_elements = last_arg_shape.elements();
                // Check if the last arg is 'sizes' input.
                // This attribute is only relevant if 'sizes' input is used.
                // The shape constraints for 'sizes' are below:
                if(last_arg_shape.type() == shape::int64_type and
                   (last_arg_elements == args.front()->get_shape().ndim() or
                    (is_axes_used() and last_arg_elements == r_attr.axes.size())))
                {
                    MIGRAPHX_THROW("PARSE_RESIZE: keep_aspect_ratio_policy is not supported!");
                }
            }
        }

        // "scales" is an attribute of the deprecated Upsample op. ver7 only
        void set_scales(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "scales"))
            {
                copy(attr.at("scales").floats(), std::back_inserter(r_attr.scales));
                vec_scale = r_attr.scales;
                compute_output_sizes();
            }
        }

        bool is_arg_skipped(const argument& arg) const { return arg.empty(); }

        bool is_arg_invalid(const instruction_ref arg) const
        {
            if(arg->name() == "undefined")
                return true;

            // skip any empty input (some of the Onnx args. are optional)
            auto lens = arg->get_shape().lens();
            return lens.empty();
        }

        void check_output_size() const
        {
            if(out_lens.size() != in_lens.size())
            {
                MIGRAPHX_THROW(
                    "PARSE_RESIZE: specified output size's rank does not match input size");
            }
        }

        void assign_output_sizes(const argument& arg_out)
        {
            arg_out.visit([&](const auto& ol) { out_lens.assign(ol.begin(), ol.end()); });
        }

        void assign_scales(const argument& arg_out)
        {
            arg_out.visit([&](const auto& v) { vec_scale.assign(v.begin(), v.end()); });
        }

        bool is_scale_rank_valid(const instruction_ref arg) const
        {
            return arg->get_shape().lens().at(0) == in_lens.size();
        }
    };

    static instruction_ref handle_nearest_neighbor(const onnx_parser::node_info& info,
                                                   resize_args& resize,
                                                   instruction_ref args_0)
    {
        // If scales are constant and input is static, emit 1-input resize with attributes
        if(resize.is_constant_scale_input() and not args_0->get_shape().dynamic())
        {
            return info.add_instruction(
                make_op("resize",
                        {{"scales", resize.vec_scale},
                         {"nearest_mode", resize.get_nearest_mode()},
                         {"coordinate_transformation_mode", resize.get_coord_trans_mode()}}),
                args_0);
        }
        // Otherwise emit 2-input resize for dynamic case
        return info.add_instruction(
            make_op("resize",
                    {{"nearest_mode", resize.get_nearest_mode()},
                     {"coordinate_transformation_mode", resize.get_coord_trans_mode()}}),
            args_0,
            resize.get_scales_sizes_arg());
    }

    static instruction_ref handle_linear_mode(const op_desc&,
                                              const onnx_parser::node_info& info,
                                              resize_args& resize,
                                              instruction_ref& args_0)
    {
        // If scales are constant and input is static, emit 1-input resize with attributes
        if(resize.is_constant_scale_input() and not args_0->get_shape().dynamic())
        {
            return info.add_instruction(
                make_op("resize",
                        {{"scales", resize.vec_scale},
                         {"mode", resize.get_mode()},
                         {"coordinate_transformation_mode", resize.get_coord_trans_mode()}}),
                args_0);
        }
        // Otherwise emit 2-input resize for dynamic case
        return info.add_instruction(
            make_op("resize",
                    {{"mode", resize.get_mode()},
                     {"coordinate_transformation_mode", resize.get_coord_trans_mode()}}),
            args_0,
            resize.get_scales_sizes_arg());
    }

    static void set_resize_attributes(const onnx_parser::node_info& info,
                                      const std::vector<instruction_ref>& args,
                                      resize_args& resize)
    {
        resize.set_coord_trans_mode(info.attributes);
        resize.set_cubic_coeff(info.attributes);
        resize.set_axes(info.attributes);
        resize.set_exclude_outside(info.attributes);
        resize.set_extrapolation_val(info.attributes);
        resize.set_aspect_ratio_policy(info.attributes, args);
        resize.set_nearest_mode(info.attributes);
        resize.set_mode(info.attributes);
    }

    static void set_resize_args(const std::vector<instruction_ref>& args, resize_args& resize)
    {
        resize.x = args.at(0);
        resize.assign_scale_or_size(args);
    }

    static void set_upsample_attributes(const onnx_parser::node_info& info, resize_args& resize)
    {
        resize.set_mode(info.attributes);
        resize.set_scales(info.attributes);
    }

    static void set_upsample_args(const std::vector<instruction_ref>& args, resize_args& resize)
    {
        resize.x = args.at(0);

        // scale is input it must be a required input
        if(not resize.is_scale_attr())
            resize.assign_scale_or_size(args);
    }

    // Split of what we handle since this parser is used for both resize/upscale operators
    static resize_args handle_inputs(const op_desc& opd,
                                     const onnx_parser::node_info& info,
                                     const std::vector<instruction_ref>& args)
    {
        resize_args resize;

        // input data shape info
        resize.in_s    = args[0]->get_shape().to_static(1);
        resize.in_lens = resize.in_s.lens();

        // output shape is explicitly specified
        resize.out_lens = std::vector<size_t>(resize.in_lens.size());

        if(opd.op_name == "upsample")
        {
            set_upsample_attributes(info, resize);
            set_upsample_args(args, resize);

            // For Upsample ver7 where scales is an attribute, create a literal for scales
            if(resize.is_scale_attr())
            {
                shape ss{shape::float_type, {resize.vec_scale.size()}};
                resize.scales_sizes_arg = info.add_literal(literal(ss, resize.vec_scale));
            }
        }
        else
        {
            set_resize_attributes(info, args, resize);
            set_resize_args(args, resize);
        }

        return resize;
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser&,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto resize = handle_inputs(opd, info, args);

        if(resize.get_mode() == "nearest")
        {
            return handle_nearest_neighbor(info, resize, args[0]);
        }
        // linear mode
        else
        {
            return handle_linear_mode(opd, info, resize, args[0]);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
