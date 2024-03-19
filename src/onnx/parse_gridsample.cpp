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
    auto one_l =
        info.add_literal(migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {1.0f}});
    auto unnorm = info.add_common_op("add", coords_t, one_l);
    if(align_corners)
    {
        // unnorm_x = (x + 1) * (size - 1) / 2
        auto mul_const = info.add_literal(
            migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {(size - 1) / 2}});
        unnorm = info.add_common_op("mul", unnorm, mul_const);
    }
    else
    {
        // unnorm_x = -0.5 + (x + 1) * size / 2
        auto mul_const = info.add_literal(
            migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {size / 2}});
        unnorm          = info.add_common_op("mul", unnorm, mul_const);
        auto minus_half = info.add_literal(
            migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {-0.5f}});
        unnorm = info.add_common_op("add", unnorm, minus_half);
    }
    return unnorm;
}

instruction_ref
clamp_values(const onnx_parser::node_info& info, instruction_ref coords_t, float max)
{
    auto min_l =
        info.add_literal(migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {0}});
    auto max_l =
        info.add_literal(migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {max}});
    return info.add_common_op("clip", coords_t, min_l, max_l);
}

instruction_ref get_sample(const onnx_parser::node_info& info,
                           instruction_ref data,
                           instruction_ref h,
                           instruction_ref w,
                           size_t n,
                           size_t c,
                           float h_max,
                           float w_max)
{
    auto nc_shape = migraphx::shape{migraphx::shape::int64_type, {2}};
    auto nc       = info.add_literal(migraphx::literal{nc_shape, {n, c}});
    auto h_clamp  = clamp_values(info, h, h_max - 1);
    auto w_clamp  = clamp_values(info, w, w_max - 1);
    auto nchw     = info.add_instruction(make_op("concat", {{"axis", 0}}), nc, h_clamp, w_clamp);
    auto sample   = info.add_instruction(make_op("gathernd"), data, nchw);
    auto h_valid  = info.add_common_op("equal", h, h_clamp);
    auto w_valid  = info.add_common_op("equal", w, w_clamp);
    auto zero     = info.add_literal(migraphx::literal{migraphx::shape{sample->get_shape()}, {0}});
    sample        = info.add_instruction(make_op("where"), h_valid, sample, zero);
    sample        = info.add_instruction(make_op("where"), w_valid, sample, zero);
    return sample;
}

instruction_ref linear_sample(const onnx_parser::node_info& info,
                              instruction_ref x0,
                              instruction_ref y0,
                              instruction_ref x1,
                              instruction_ref y1,
                              instruction_ref w1,
                              instruction_ref w2,
                              instruction_ref w3,
                              instruction_ref w4,
                              instruction_ref data,
                              size_t n,
                              size_t c,
                              float h_max,
                              float w_max)
{
    auto get_weighted_pixel = [&](auto y, auto x, auto w) {
        auto p = get_sample(info, data, y, x, n, c, h_max, w_max);
        return info.add_common_op("mul", p, w);
    };

    auto p1 = get_weighted_pixel(y0, x0, w1);
    auto p2 = get_weighted_pixel(y0, x1, w2);
    auto p3 = get_weighted_pixel(y1, x0, w3);
    auto p4 = get_weighted_pixel(y1, x1, w4);

    auto res = info.add_common_op("add", p1, p2);
    res      = info.add_common_op("add", res, p3);
    return info.add_common_op("add", res, p4);
}

struct grid_sampler
{
    std::string m_padding;
    bool m_align_corners;

    instruction_ref m_input;
    instruction_ref m_grid;

    size_t m_batch;
    size_t m_channel;
    size_t m_in_height;
    size_t m_in_width;
    size_t m_out_height;
    size_t m_out_width;

    instruction_ref m_one_l;
    instruction_ref m_zero_l;
    instruction_ref m_minus_half_l;
    instruction_ref m_width_l;
    instruction_ref m_height_l;
    instruction_ref m_unnorm_x;
    instruction_ref m_unnorm_y;

    grid_sampler(instruction_ref&& input, instruction_ref&& grid, bool align, std::string padding)
        : m_padding(padding), m_align_corners(align), m_input(input), m_grid(grid)
    {
        auto i_lens  = input->get_shape().lens();
        m_batch      = i_lens.at(0);
        m_channel    = i_lens.at(1);
        m_in_height  = i_lens.at(2);
        m_in_width   = i_lens.at(3);
        auto g_lens  = grid->get_shape().lens();
        m_out_height = g_lens.at(1);
        m_out_width  = g_lens.at(2);
    }

    virtual ~grid_sampler() {}

    virtual void setup(const onnx_parser::node_info& info)
    {
        m_one_l = info.add_literal(
            migraphx::literal{migraphx::shape{m_input->get_shape().type()}, {1.0f}});
        m_zero_l = info.add_literal(
            migraphx::literal{migraphx::shape{m_input->get_shape().type()}, {0.0f}});
        m_minus_half_l = info.add_literal(
            migraphx::literal{migraphx::shape{m_input->get_shape().type()}, {-0.5f}});
        m_width_l = info.add_literal(
            migraphx::literal{migraphx::shape{m_input->get_shape().type()}, {m_in_width - 1}});
        m_height_l = info.add_literal(
            migraphx::literal{migraphx::shape{m_input->get_shape().type()}, {m_in_height - 1}});

        auto x_coords = info.add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), m_grid);

        auto y_coords = info.add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}), m_grid);

        x_coords   = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), x_coords);
        y_coords   = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), y_coords);
        m_unnorm_x = unnormalize(info, x_coords, m_in_width);
        m_unnorm_y = unnormalize(info, y_coords, m_in_height);

        if(m_padding == "border")
        {
            m_unnorm_x = info.add_common_op("clip", m_unnorm_x, m_zero_l, m_width_l);
            m_unnorm_y = info.add_common_op("clip", m_unnorm_y, m_zero_l, m_height_l);
        }
    }

    instruction_ref
    unnormalize(const onnx_parser::node_info& info, const instruction_ref& coords_t, float size)
    {
        auto unnorm = info.add_common_op("add", coords_t, m_one_l);
        if(m_align_corners)
        {
            // unnorm_x = (x + 1) * (size - 1) / 2
            auto mul_const = info.add_literal(
                migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {(size - 1) / 2}});
            unnorm = info.add_common_op("mul", unnorm, mul_const);
        }
        else
        {
            // unnorm_x = -0.5 + (x + 1) * size / 2
            auto mul_const = info.add_literal(
                migraphx::literal{migraphx::shape{coords_t->get_shape().type()}, {size / 2}});
            unnorm = info.add_common_op("mul", unnorm, mul_const);
            unnorm = info.add_common_op("add", unnorm, m_minus_half_l);
        }
        return unnorm;
    }

    inline bool has_border_padding() const { return m_padding == "border"; }
};

struct nearest_sampler : grid_sampler
{
    instruction_ref m_round_x;
    instruction_ref m_round_y;

    nearest_sampler(instruction_ref&& input,
                    instruction_ref&& grid,
                    bool align,
                    std::string padding)
        : grid_sampler(std::move(input), std::move(grid), align, padding)
    {
    }

    void setup(const onnx_parser::node_info& info) override
    {
        grid_sampler::setup(info);
        m_round_x = info.add_common_op("nearbyint", m_unnorm_x);
        m_round_y = info.add_common_op("nearbyint", m_unnorm_y);
    }

    void update_indices(const onnx_parser::node_info& info,
                        const instruction_ref& h,
                        const instruction_ref& w,
                        size_t n,
                        size_t c,
                        std::vector<instruction_ref>& indices,
                        std::vector<instruction_ref>& validation)
    {
        static auto nc_shape = migraphx::shape{m_input->get_shape().type(), {2}};
        auto nc              = info.add_literal(migraphx::literal{nc_shape, {n, c}});
        auto w_clamp =
            has_border_padding() ? w : info.add_common_op("clip", w, m_zero_l, m_width_l);
        auto h_clamp =
            has_border_padding() ? h : info.add_common_op("clip", h, m_zero_l, m_height_l);
        auto nchw = info.add_instruction(make_op("concat", {{"axis", 0}}), nc, h_clamp, w_clamp);
        indices.push_back(nchw);
        if(not has_border_padding())
        {
            auto h_valid = info.add_common_op("equal", h, h_clamp);
            auto w_valid = info.add_common_op("equal", w, w_clamp);
            auto valid   = info.add_common_op("logical_and", h_valid, w_valid);
            validation.push_back(valid);
        }
    }

    instruction_ref sample(const onnx_parser::node_info& info)
    {
        setup(info);
        std::vector<instruction_ref> indices;
        std::vector<instruction_ref> validation;
        static auto nhw_shape = migraphx::shape{migraphx::shape::int64_type, {3}};
        for(size_t n = 0; n < m_batch; n++)
        {
            for(size_t h = 0; h < m_out_height; h++)
            {
                for(size_t w = 0; w < m_out_width; w++)
                {
                    auto nhw = info.add_literal(migraphx::literal{nhw_shape, {n, h, w}});
                    auto h_t = info.add_instruction(make_op("gathernd"), m_round_y, nhw);
                    auto w_t = info.add_instruction(make_op("gathernd"), m_round_x, nhw);
                    for(size_t c = 0; c < m_channel; c++)
                    {
                        update_indices(info, h_t, w_t, n, c, indices, validation);
                    }
                }
            }
        }

        auto indices_t = indices.at(0);
        std::for_each(std::next(indices.begin()), indices.end(), [&info, &indices_t](auto& index) {
            indices_t = info.add_instruction(make_op("concat", {{"axis", 0}}), indices_t, index);
        });
        indices_t = info.add_instruction(
            make_op("reshape", {{"dims", {indices_t->get_shape().elements() / 4, 4}}}), indices_t);
        auto samples = info.add_instruction(make_op("gathernd"), m_input, indices_t);
        if(not has_border_padding())
        {
            auto validation_t = validation.at(0);
            std::for_each(std::next(validation.begin()),
                          validation.end(),
                          [&info, &validation_t](auto& valid) {
                              validation_t = info.add_instruction(
                                  make_op("concat", {{"axis", 0}}), validation_t, valid);
                          });
            samples = info.add_common_op("where", validation_t, samples, m_zero_l);
        }

        samples = info.add_instruction(
            make_op("reshape", {{"dims", {m_batch, m_out_height, m_out_width, m_channel}}}),
            samples);
        samples =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), samples);
        return samples;
    }
};

struct parse_gridsample : op_parser<parse_gridsample>
{
    std::vector<op_desc> operators() const { return {{"GridSample"}}; }
    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        bool align_corners       = false;
        std::string mode         = "linear";
        std::string padding_mode = "zeros";

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
        if(x_dims != 4)
        {
            MIGRAPHX_THROW("PARSE_GRID_SAMPLE: only 4-D inputs are supported");
        }

        // parse 4-D
        // x: [batch, channel, in_height, in_width]
        auto batch     = x_lens.at(0);
        auto channel   = x_lens.at(1);
        auto in_height = x_lens.at(2);
        auto in_width  = x_lens.at(3);
        // grid: [batch, out_height, out_width, 2]
        auto out_height = grid_shape.lens().at(1);
        auto out_width  = grid_shape.lens().at(2);

        // unnormalize grid coordinates
        auto x_coords = info.add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), grid);

        auto y_coords = info.add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}), grid);

        x_coords      = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), x_coords);
        y_coords      = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), y_coords);
        auto unnorm_x = unnomralize_grid_cord(info, x_coords, in_width, align_corners);
        auto unnorm_y = unnomralize_grid_cord(info, y_coords, in_height, align_corners);

        if(padding_mode == "border")
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

        std::unordered_map<std::string, instruction_ref> floor_x_cache;
        std::unordered_map<std::string, instruction_ref> floor_y_cache;
        std::unordered_map<std::string, instruction_ref> ceil_x_cache;
        std::unordered_map<std::string, instruction_ref> ceil_y_cache;
        std::unordered_map<std::string, instruction_ref> wa_cache;
        std::unordered_map<std::string, instruction_ref> wb_cache;
        std::unordered_map<std::string, instruction_ref> wc_cache;
        std::unordered_map<std::string, instruction_ref> wd_cache;

        if(contains(mode, "linear"))
        {
            floor_x  = info.add_common_op("floor", unnorm_x);
            floor_y  = info.add_common_op("floor", unnorm_y);
            auto one = info.add_literal(
                migraphx::literal{migraphx::shape{grid->get_shape().type()}, {1.0f}});
            ceil_x                 = info.add_common_op("add", floor_x, one);
            ceil_y                 = info.add_common_op("add", floor_y, one);
            auto fract_x           = info.add_common_op("sub", unnorm_x, floor_x);
            auto fract_y           = info.add_common_op("sub", unnorm_y, floor_y);
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
        std::unordered_map<std::string, instruction_ref> round_x_cache;
        std::unordered_map<std::string, instruction_ref> round_y_cache;

        if(mode == "nearest")
        {
            auto sampler =
                nearest_sampler(std::move(x), std::move(grid), align_corners, padding_mode);
            return sampler.sample(info);
        }

        std::vector<instruction_ref> samples;
        for(size_t n = 0; n < batch; n++)
        {
            for(size_t h = 0; h < out_height; h++)
            {
                for(size_t w = 0; w < out_width; w++)
                {
                    std::stringstream ss;
                    ss << n << "_" << h << "_" << w;
                    auto nhw_key = ss.str();

                    auto nhw_shape = migraphx::shape{migraphx::shape::int64_type, {3}};
                    auto nhw       = info.add_literal(migraphx::literal{nhw_shape, {n, h, w}});

                    auto y0 = info.add_instruction(make_op("gathernd"), floor_y, nhw);
                    floor_y_cache[nhw_key] = info.add_instruction(
                        make_op("convert", {{"target_type", migraphx::shape::int64_type}}), y0);

                    auto x0 = info.add_instruction(make_op("gathernd"), floor_x, nhw);
                    floor_x_cache[nhw_key] = info.add_instruction(
                        make_op("convert", {{"target_type", migraphx::shape::int64_type}}), x0);

                    auto y1               = info.add_instruction(make_op("gathernd"), ceil_y, nhw);
                    ceil_y_cache[nhw_key] = info.add_instruction(
                        make_op("convert", {{"target_type", migraphx::shape::int64_type}}), y1);
                    auto x1               = info.add_instruction(make_op("gathernd"), ceil_x, nhw);
                    ceil_x_cache[nhw_key] = info.add_instruction(
                        make_op("convert", {{"target_type", migraphx::shape::int64_type}}), x1);

                    wa_cache[nhw_key] = info.add_instruction(make_op("gathernd"), wa, nhw);
                    wb_cache[nhw_key] = info.add_instruction(make_op("gathernd"), wb, nhw);
                    wc_cache[nhw_key] = info.add_instruction(make_op("gathernd"), wc, nhw);
                    wd_cache[nhw_key] = info.add_instruction(make_op("gathernd"), wd, nhw);
                    for(size_t c = 0; c < channel; c++)
                    {
                        samples.push_back(linear_sample(info,
                                                        floor_x_cache.at(nhw_key),
                                                        floor_y_cache.at(nhw_key),
                                                        ceil_x_cache.at(nhw_key),
                                                        ceil_y_cache.at(nhw_key),
                                                        wa_cache.at(nhw_key),
                                                        wb_cache.at(nhw_key),
                                                        wc_cache.at(nhw_key),
                                                        wd_cache.at(nhw_key),
                                                        x,
                                                        n,
                                                        c,
                                                        in_height,
                                                        in_width));
                    }
                }
            }
        }

        auto output = samples.at(0);
        for(size_t i = 1; i < samples.size(); ++i)
        {
            output = info.add_instruction(make_op("concat", {{"axis", 0}}), output, samples.at(i));
        }
        output = info.add_instruction(
            make_op("reshape", {{"dims", {batch, out_height, out_width, channel}}}), output);
        output =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), output);
        return output;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
