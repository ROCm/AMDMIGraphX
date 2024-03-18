/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

instruction_ref unnomralize_grid_cord(const onnx_parser::node_info& info,
                                      instruction_ref coords_t,
                                      float size,
                                      bool align_corners)
{
    auto one_l = info.add_literal(migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {1.0f}});
    auto unnorm = info.add_common_op("add", coords_t, one_l);
    if (align_corners)
    {
        // unnorm_x = (x + 1) * (size - 1) / 2
        auto mul_const    = info.add_literal(
            migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {(size - 1) / 2}});
        unnorm = info.add_common_op("mul", unnorm, mul_const);
    }
    else
    {
        // unnorm_x = -0.5 + (x + 1) * size / 2
        auto mul_const    = info.add_literal(
            migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {size / 2}});
        unnorm = info.add_common_op("mul", unnorm, mul_const);
        auto minus_half = info.add_literal(migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {-0.5f}});
        unnorm = info.add_common_op("add", unnorm, minus_half);
    }
    return unnorm;
}

instruction_ref clamp_values(const onnx_parser::node_info& info,
                                      instruction_ref coords_t,
                                      float max)
{
    auto min_l = info.add_literal(migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {0}});
    auto max_l = info.add_literal(migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {max}});
    return info.add_common_op("clip", coords_t, min_l, max_l);
}

instruction_ref get_pixel(const onnx_parser::node_info& info,
                                instruction_ref data,
                                instruction_ref h,
                                instruction_ref w,
                                size_t n,
                                size_t c,
                                float h_max,
                                float w_max)
{
    auto nc_shape = migraphx::shape{migraphx::shape::int64_type, {2}};
    auto nc = info.add_literal(migraphx::literal{nc_shape, {n, c}});
    auto h_clamp = clamp_values(info, h, h_max - 1);
    auto w_clamp = clamp_values(info, w, w_max - 1);
    auto nchw = info.add_instruction(make_op("concat", {{"axis", 0}}), nc, h_clamp, w_clamp);
    auto pixels = info.add_instruction(make_op("gathernd"), data, nchw);
    auto h_valid = info.add_common_op("equal", h, h_clamp);
    auto w_valid = info.add_common_op("equal", w, w_clamp);
    auto zero     = info.add_literal(migraphx::literal{migraphx::shape{pixels->get_shape()}, {0}});
    pixels = info.add_instruction(make_op("where"), h_valid, pixels, zero);
    pixels = info.add_instruction(make_op("where"), w_valid, pixels, zero);
    return pixels;
}

instruction_ref nearest_sample(const onnx_parser::node_info& info,
                                instruction_ref round_x,
                                instruction_ref round_y,
                                instruction_ref data,
                                std::vector<size_t> nchw_values,
                                float h_max,
                                float w_max)
{
    auto nhw_shape = migraphx::shape{migraphx::shape::int64_type, {3}};
    auto nhw = info.add_literal(migraphx::literal{nhw_shape, {nchw_values.at(0), nchw_values.at(2), nchw_values.at(3)}});
    auto h = info.add_instruction(make_op("gathernd"), round_y, nhw);
    h = info.add_instruction(make_op("convert", {{"target_type", migraphx::shape::int64_type}}), h);
    auto w = info.add_instruction(make_op("gathernd"), round_x, nhw);
    w = info.add_instruction(make_op("convert", {{"target_type", migraphx::shape::int64_type}}), w);
    return get_pixel(info, data, h, w, nchw_values.at(0), nchw_values.at(1), h_max, w_max);
}

instruction_ref linear_sample(const onnx_parser::node_info& info,
                              instruction_ref floor_x,
                              instruction_ref floor_y,
                              instruction_ref ceil_x,
                              instruction_ref ceil_y,
                              instruction_ref wa,
                              instruction_ref wb,
                              instruction_ref wc,
                              instruction_ref wd,
                              instruction_ref data,
                              std::vector<size_t> nchw_values,
                              float h_max,
                              float w_max)
{
    auto nhw_shape = migraphx::shape{migraphx::shape::int64_type, {3}};
    auto nhw       = info.add_literal(
        migraphx::literal{nhw_shape, {nchw_values.at(0), nchw_values.at(2), nchw_values.at(3)}});

    auto y0 = info.add_instruction(make_op("gathernd"), floor_y, nhw);
    y0 = info.add_instruction(make_op("convert", {{"target_type", migraphx::shape::int64_type}}), y0);
    auto x0 = info.add_instruction(make_op("gathernd"), floor_x, nhw);
    x0 = info.add_instruction(make_op("convert", {{"target_type", migraphx::shape::int64_type}}), x0);

    auto y1 = info.add_instruction(make_op("gathernd"), ceil_y, nhw);
    y1 = info.add_instruction(make_op("convert", {{"target_type", migraphx::shape::int64_type}}), y1);
    auto x1 = info.add_instruction(make_op("gathernd"), ceil_x, nhw);
    x1 = info.add_instruction(make_op("convert", {{"target_type", migraphx::shape::int64_type}}), x1);

    auto w1 = info.add_instruction(make_op("gathernd"), wa, nhw);
    auto w2 = info.add_instruction(make_op("gathernd"), wb, nhw);
    auto w3 = info.add_instruction(make_op("gathernd"), wc, nhw);
    auto w4 = info.add_instruction(make_op("gathernd"), wd, nhw);
    
    auto p1 = get_pixel(info, data, y0, x0, nchw_values.at(0), nchw_values.at(1), h_max, w_max);
    p1 = info.add_common_op("mul", p1, w1);

    auto p2 = get_pixel(info, data, y0, x1, nchw_values.at(0), nchw_values.at(1), h_max, w_max);
    p2 = info.add_common_op("mul", p2, w2);

    auto p3 = get_pixel(info, data, y1, x0, nchw_values.at(0), nchw_values.at(1), h_max, w_max);
    p3 = info.add_common_op("mul", p3, w3);

    auto p4 = get_pixel(info, data, y1, x1, nchw_values.at(0), nchw_values.at(1), h_max, w_max);
    p4 = info.add_common_op("mul", p4, w4);

    auto res = info.add_common_op("add", p1, p2);
    res = info.add_common_op("add", res, p3);
    return info.add_common_op("add", res, p4);
}

struct parse_gridsample : op_parser<parse_gridsample>
{
    std::vector<op_desc> operators() const { return {{"GridSample"}}; }
    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        bool align_corners         = false;
        std::string mode           = "linear";
        std::string padding_mode   = "zeros";

        if(contains(info.attributes, "align_corners"))
        {
            align_corners = parser.parse_value(info.attributes.at("align_corners")).at<bool>();
        }

        if(contains(info.attributes, "mode"))
        {
            mode = info.attributes.at("mode").s();
            if(contains(mode, "cubic"))
            {
                MIGRAPHX_THROW("PARSE_GRID_SAMPLE: cubic mode is not supported");
            }
        }

        if(contains(info.attributes, "padding_mode"))
        {
            padding_mode = info.attributes.at("padding_mode").s();
            if(padding_mode == "reflection")
            {
                MIGRAPHX_THROW("PARSE_GRID_SAMPLE: reflect padding_mode is not supported");
            }
        }

        auto grid       = args.at(1);
        auto grid_shape = grid->get_shape();
        if(not is_type_float(grid_shape.type()))
        {
            MIGRAPHX_THROW("PARSE_GRID_SAMPLE: grid input must have floating type");
        }
        auto x       = args.at(0);
        auto x_shape = x->get_shape();
        auto x_lens  = x_shape.lens();
        auto x_dims  = x_lens.size();
        if(grid_shape.lens().size() != x_dims)
        {
            MIGRAPHX_THROW(
                "PARSE_GRID_SAMPLE: x and grid inputs must have same number of dimensions");
        }
        if (x_dims != 4)
        {
            MIGRAPHX_THROW("PARSE_GRID_SAMPLE: only 4-D inputs are supported");
        }

        // parse 4-D
        // x: [batch, channel, in_height, in_width]
        auto batch    = x_lens.at(0);
        auto channel   = x_lens.at(1);
        auto in_height  = x_lens.at(2);
        auto in_width   = x_lens.at(3);
        // grid: [batch, out_height, out_width, 2]
        auto out_height = grid_shape.lens().at(1);
        auto out_width  = grid_shape.lens().at(2);

        // unnormalize grid coordinates
        auto x_coords = info.add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}),
            grid);

        auto y_coords = info.add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}),
            grid);


        x_coords      = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), x_coords);
        y_coords      = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), y_coords);
        auto unnorm_x = unnomralize_grid_cord(info, x_coords, in_width, align_corners);
        auto unnorm_y = unnomralize_grid_cord(info, y_coords, in_height, align_corners);

        if (padding_mode == "border")
        {
            unnorm_x = clamp_values(info, unnorm_x, in_width - 1);
            unnorm_y = clamp_values(info, unnorm_y, in_height - 1);
        }

        // floor, ceil and weight values required for linear interpolation
        instruction_ref floor_x;
        instruction_ref floor_y;
        instruction_ref ceil_x;
        instruction_ref ceil_y;
        instruction_ref wa;
        instruction_ref wb;
        instruction_ref wc;
        instruction_ref wd;

        if(contains(mode,"linear"))
        {
            floor_x = info.add_common_op("floor", unnorm_x);
            floor_y = info.add_common_op("floor", unnorm_y);
            auto one = info.add_literal(migraphx::literal{migraphx::shape{grid->get_shape().type()}, {1.0f}});
            ceil_x = info.add_common_op("add", floor_x, one);
            ceil_y = info.add_common_op("add", floor_y, one);
            auto fract_x = info.add_common_op("sub", unnorm_x, floor_x);
            auto fract_y = info.add_common_op("sub", unnorm_y, floor_y);
            auto one_minus_fract_x = info.add_common_op("sub", one, fract_x);
            auto one_minus_fract_y = info.add_common_op("sub", one, fract_y);
            wa = info.add_common_op("mul", one_minus_fract_y, one_minus_fract_x);
            wb = info.add_common_op("mul", one_minus_fract_y, fract_x);
            wc = info.add_common_op("mul", fract_y, one_minus_fract_x);
            wd = info.add_common_op("mul", fract_y, fract_x);
        }

        // rounded values required for nearest interpolation
        instruction_ref round_x;
        instruction_ref round_y;

        if (mode == "nearest")
        {
            round_x = info.add_common_op("nearbyint", unnorm_x);
            round_y = info.add_common_op("nearbyint", unnorm_y);
        }

        std::vector<instruction_ref> pixels;
        for (size_t n = 0; n < batch; n++)
        {
            for (size_t c = 0; c < channel; c++)
            {
                for (size_t h = 0; h < out_height; h++)
                {
                    for (size_t w = 0; w <out_width; w++)
                    {
                        if (mode == "nearest")
                        {
                            pixels.push_back(nearest_sample(info, round_x, round_y, x, {n, c, h, w}, in_height, in_width));
                        }
                        // linear
                        else
                        {
                            pixels.push_back(linear_sample(info, floor_x, floor_y, ceil_x, ceil_y, wa, wb, wc, wd, x, {n, c, h, w}, in_height, in_width));
                        }
                    }
                }
            }
        }
        
        auto output = pixels.at(0);
        for (size_t i = 1; i < pixels.size(); ++i)
        {
            output = info.add_instruction(make_op("concat", {{"axis", 0}}), output, pixels.at(i));
        }
        output = info.add_instruction(make_op("reshape", {{"dims", {batch, channel, out_height, out_width}}}), output);
        return output;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
