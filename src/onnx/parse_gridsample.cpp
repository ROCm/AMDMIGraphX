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

struct grid_sampler
{
    std::string m_padding;
    bool m_align_corners;

    instruction_ref m_input;
    instruction_ref m_grid;

    size_t m_batch{1};
    size_t m_channel{1};
    size_t m_in_height{1};
    size_t m_in_width{1};
    size_t m_out_height{1};
    size_t m_out_width{1};

    instruction_ref m_one_l;
    instruction_ref m_two_l;
    instruction_ref m_zero_l;
    instruction_ref m_minus_half_l;
    instruction_ref m_width_l;
    instruction_ref m_width_max_l;
    instruction_ref m_height_l;
    instruction_ref m_height_max_l;
    instruction_ref m_unnorm_x;
    instruction_ref m_unnorm_y;

    grid_sampler(const instruction_ref& input,
                 const instruction_ref& grid,
                 bool align,
                 std::string&& padding)
        : m_padding(std::move(padding)), m_align_corners(align), m_input(input), m_grid(grid)
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

    instruction_ref reflect_coordinates(const onnx_parser::node_info& info,
                                        instruction_ref coords,
                                        instruction_ref size,
                                        instruction_ref corner_start) const
    {
        // TODO: add reduce_min and if operator check
        auto index_align_corner = info.add_common_op("sub", corner_start, coords);
        index_align_corner      = info.add_common_op("abs", index_align_corner);
        auto size_times         = info.add_instruction(
            migraphx::make_op("convert",
                                      {{"target_type", migraphx::to_value(migraphx::shape::int64_type)}}),
            index_align_corner);
        size_times = info.add_common_op("div", size_times, size);
        size_times = info.add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int64_type)}}),
            size_times);
        auto cond   = info.add_common_op("mod", size_times, m_two_l);
        cond        = info.add_common_op("equal", cond, m_zero_l);
        auto extra  = info.add_common_op("mul", size_times, size);
        extra       = info.add_common_op("sub", index_align_corner, extra);
        auto cond_a = info.add_common_op("add", extra, corner_start);
        auto cond_b = info.add_common_op("sub", size, extra);
        cond_b      = info.add_common_op("add", cond_b, corner_start);
        return info.add_common_op("where", cond, cond_a, cond_b);
    }

    virtual void setup(const onnx_parser::node_info& info)
    {
        auto type      = m_grid->get_shape().type();
        m_zero_l       = info.add_literal(migraphx::literal{migraphx::shape{type}, {0.0f}});
        m_one_l        = info.add_literal(migraphx::literal{migraphx::shape{type}, {1.0f}});
        m_minus_half_l = info.add_literal(migraphx::literal{migraphx::shape{type}, {-0.5f}});
        m_width_max_l =
            info.add_literal(migraphx::literal{migraphx::shape{type}, {m_in_width - 1}});
        m_width_l = info.add_literal(migraphx::literal{migraphx::shape{type}, {m_in_width}});
        m_height_max_l =
            info.add_literal(migraphx::literal{migraphx::shape{type}, {m_in_height - 1}});
        m_height_l = info.add_literal(migraphx::literal{migraphx::shape{type}, {m_in_height}});

        auto x_coords = info.add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), m_grid);

        auto y_coords = info.add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}), m_grid);

        x_coords   = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), x_coords);
        y_coords   = info.add_instruction(make_op("squeeze", {{"axes", {3}}}), y_coords);
        m_unnorm_x = unnormalize(info, x_coords, m_in_width);
        m_unnorm_y = unnormalize(info, y_coords, m_in_height);

        if(m_padding == "reflection")
        {
            m_two_l = info.add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {2}});
            auto corner_start = m_align_corners ? m_zero_l : m_minus_half_l;
            m_unnorm_x        = reflect_coordinates(
                info, m_unnorm_x, m_align_corners ? m_width_max_l : m_width_l, corner_start);
            m_unnorm_y = reflect_coordinates(
                info, m_unnorm_y, m_align_corners ? m_height_max_l : m_height_l, corner_start);
            m_unnorm_x = info.add_common_op("clip", m_unnorm_x, m_zero_l, m_width_max_l);
            m_unnorm_y = info.add_common_op("clip", m_unnorm_y, m_zero_l, m_height_max_l);
        }

        if(m_padding == "border")
        {
            m_unnorm_x = info.add_common_op("clip", m_unnorm_x, m_zero_l, m_width_max_l);
            m_unnorm_y = info.add_common_op("clip", m_unnorm_y, m_zero_l, m_height_max_l);
        }
    }

    instruction_ref unnormalize(const onnx_parser::node_info& info,
                                const instruction_ref& coords_t,
                                float size) const
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

    void update_indices(const onnx_parser::node_info& info,
                        const instruction_ref& h,
                        const instruction_ref& w,
                        size_t n,
                        size_t c,
                        std::vector<instruction_ref>& indices,
                        std::vector<instruction_ref>& validation,
                        bool validate = true) const
    {
        static auto nc_shape = migraphx::shape{m_grid->get_shape().type(), {2}};
        auto nc              = info.add_literal(migraphx::literal{nc_shape, {n, c}});
        auto w_clamp = validate ? info.add_common_op("clip", w, m_zero_l, m_width_max_l) : w;
        auto h_clamp = validate ? info.add_common_op("clip", h, m_zero_l, m_height_max_l) : h;
        auto nchw    = info.add_instruction(make_op("concat", {{"axis", 0}}), nc, h_clamp, w_clamp);
        indices.push_back(nchw);
        if(validate)
        {
            auto h_valid = info.add_common_op("equal", h, h_clamp);
            auto w_valid = info.add_common_op("equal", w, w_clamp);
            auto valid   = info.add_common_op("logical_and", h_valid, w_valid);
            validation.push_back(valid);
        }
    }

    static instruction_ref concat_on_first_dim(const onnx_parser::node_info& info,
                                               std::vector<instruction_ref> instructions)
    {
        if(instructions.empty())
            MIGRAPHX_THROW("PARSE_GRID_SAMPLE: can't concatenate empty list of instructions");

        auto ret = instructions.at(0);
        std::for_each(
            std::next(instructions.begin()), instructions.end(), [&info, &ret](auto& ins) {
                ret = info.add_instruction(make_op("concat", {{"axis", 0}}), ret, ins);
            });
        return ret;
    }

    inline bool has_border_padding() const { return m_padding == "border"; }

    virtual instruction_ref sample(const onnx_parser::node_info& info) = 0;
};

struct nearest_sampler : grid_sampler
{
    instruction_ref m_round_x;
    instruction_ref m_round_y;

    nearest_sampler(const instruction_ref& input,
                    const instruction_ref& grid,
                    bool align,
                    std::string&& padding)
        : grid_sampler(input, grid, align, std::move(padding))
    {
    }

    void setup(const onnx_parser::node_info& info) override
    {
        grid_sampler::setup(info);
        m_round_x = info.add_common_op("nearbyint", m_unnorm_x);
        m_round_y = info.add_common_op("nearbyint", m_unnorm_y);
    }

    instruction_ref sample(const onnx_parser::node_info& info) override
    {
        setup(info);
        std::vector<instruction_ref> indices;
        std::vector<instruction_ref> validation;
        static auto nhw_shape = migraphx::shape{migraphx::shape::int64_type, {3}};
        bool validate         = not has_border_padding();
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
                        update_indices(info, h_t, w_t, n, c, indices, validation, validate);
                    }
                }
            }
        }

        auto indices_t = concat_on_first_dim(info, indices);
        indices_t      = info.add_instruction(
            make_op("reshape", {{"dims", {indices_t->get_shape().elements() / 4, 4}}}), indices_t);
        auto samples = info.add_instruction(make_op("gathernd"), m_input, indices_t);
        if(validate)
        {
            auto validation_t = concat_on_first_dim(info, validation);
            samples           = info.add_common_op("where", validation_t, samples, m_zero_l);
        }

        samples = info.add_instruction(
            make_op("reshape", {{"dims", {m_batch, m_out_height, m_out_width, m_channel}}}),
            samples);
        samples =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), samples);
        samples = info.add_instruction(
            make_op("convert", {{"target_type", m_input->get_shape().type()}}), samples);
        return samples;
    }
};

struct linear_sampler : grid_sampler
{
    instruction_ref m_floor_x;
    instruction_ref m_floor_y;
    instruction_ref m_ceil_x;
    instruction_ref m_ceil_y;
    std::array<instruction_ref, 4> m_corner_weights;

    linear_sampler(const instruction_ref& input,
                   const instruction_ref& grid,
                   bool align,
                   std::string&& padding)
        : grid_sampler(input, grid, align, std::move(padding))
    {
    }

    void setup(const onnx_parser::node_info& info) override
    {
        grid_sampler::setup(info);
        m_floor_x              = info.add_common_op("floor", m_unnorm_x);
        m_floor_y              = info.add_common_op("floor", m_unnorm_y);
        m_ceil_x               = info.add_common_op("add", m_floor_x, m_one_l);
        m_ceil_y               = info.add_common_op("add", m_floor_y, m_one_l);
        auto fract_x           = info.add_common_op("sub", m_unnorm_x, m_floor_x);
        auto fract_y           = info.add_common_op("sub", m_unnorm_y, m_floor_y);
        auto one_minus_fract_x = info.add_common_op("sub", m_one_l, fract_x);
        auto one_minus_fract_y = info.add_common_op("sub", m_one_l, fract_y);
        m_corner_weights[0]    = info.add_common_op("mul", one_minus_fract_y, one_minus_fract_x);
        m_corner_weights[1]    = info.add_common_op("mul", one_minus_fract_y, fract_x);
        m_corner_weights[2]    = info.add_common_op("mul", fract_y, one_minus_fract_x);
        m_corner_weights[3]    = info.add_common_op("mul", fract_y, fract_x);
    }

    instruction_ref sample(const onnx_parser::node_info& info) override
    {
        setup(info);
        std::array<std::vector<instruction_ref>, 4> indices_all;
        std::array<std::vector<instruction_ref>, 4> validation_all;
        std::vector<instruction_ref> weight_indices;

        static auto nhw_shape = migraphx::shape{migraphx::shape::int64_type, {3}};
        for(size_t n = 0; n < m_batch; n++)
        {
            for(size_t h = 0; h < m_out_height; h++)
            {
                for(size_t w = 0; w < m_out_width; w++)
                {
                    auto nhw = info.add_literal(migraphx::literal{nhw_shape, {n, h, w}});
                    auto y0  = info.add_instruction(make_op("gathernd"), m_floor_y, nhw);
                    auto x0  = info.add_instruction(make_op("gathernd"), m_floor_x, nhw);
                    auto y1  = info.add_instruction(make_op("gathernd"), m_ceil_y, nhw);
                    auto x1  = info.add_instruction(make_op("gathernd"), m_ceil_x, nhw);
                    weight_indices.push_back(nhw);
                    for(size_t c = 0; c < m_channel; c++)
                    {
                        update_indices(info, y0, x0, n, c, indices_all.at(0), validation_all.at(0));
                        update_indices(info, y0, x1, n, c, indices_all.at(1), validation_all.at(1));
                        update_indices(info, y1, x0, n, c, indices_all.at(2), validation_all.at(2));
                        update_indices(info, y1, x1, n, c, indices_all.at(3), validation_all.at(3));
                    }
                }
            }
        }

        std::vector<instruction_ref> weighted_corners;
        auto weight_index_t = concat_on_first_dim(info, weight_indices);
        weight_index_t      = info.add_instruction(
            make_op("reshape", {{"dims", {weight_indices.size(), 3}}}), weight_index_t);
        for(auto i = 0; i < 4; ++i)
        {
            auto indices    = indices_all.at(i);
            auto validation = validation_all.at(i);
            auto indices_t  = concat_on_first_dim(info, indices);
            indices_t       = info.add_instruction(
                make_op("reshape", {{"dims", {indices_t->get_shape().elements() / 4, 4}}}),
                indices_t);
            auto samples      = info.add_instruction(make_op("gathernd"), m_input, indices_t);
            auto validation_t = concat_on_first_dim(info, validation);
            samples           = info.add_common_op("where", validation_t, samples, m_zero_l);
            auto weights =
                info.add_instruction(make_op("gathernd"), m_corner_weights.at(i), weight_index_t);
            weighted_corners.push_back(info.add_instruction(make_op("mul"), samples, weights));
        }

        auto samples = weighted_corners.at(0);
        std::for_each(std::next(weighted_corners.begin()),
                      weighted_corners.end(),
                      [&info, &samples](auto& s) {
                          samples = info.add_instruction(make_op("add"), samples, s);
                      });
        samples = info.add_instruction(
            make_op("reshape", {{"dims", {m_batch, m_out_height, m_out_width, m_channel}}}),
            samples);
        samples =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), samples);
        samples = info.add_instruction(
            make_op("convert", {{"target_type", m_input->get_shape().type()}}), samples);
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
        bool align_corners = false;
        // Note: defult mode can be linear or bilinear depending on the onnx version
        std::string mode         = "linear";
        std::string padding_mode = "zeros";

        if(contains(info.attributes, "align_corners"))
        {
            align_corners = parser.parse_value(info.attributes.at("align_corners")).at<bool>();
        }

        if(contains(info.attributes, "mode"))
        {
            mode = info.attributes.at("mode").s();
            // Note: can be cubic or bicubic depending on the onnx version
            if(contains(mode, "cubic"))
            {
                MIGRAPHX_THROW("PARSE_GRID_SAMPLE: cubic mode is not supported");
            }
        }

        if(contains(info.attributes, "padding_mode"))
        {
            padding_mode = info.attributes.at("padding_mode").s();
        }

        const auto& grid       = args.at(1);
        const auto& grid_shape = grid->get_shape();
        if(not is_type_float(grid_shape.type()))
        {
            MIGRAPHX_THROW("PARSE_GRID_SAMPLE: grid input must have floating type");
        }
        const auto& x      = args.at(0);
        const auto& x_dims = x->get_shape().lens().size();
        if(grid_shape.lens().size() != x_dims)
        {
            MIGRAPHX_THROW(
                "PARSE_GRID_SAMPLE: x and grid inputs must have same number of dimensions");
        }
        if(x_dims != 4)
        {
            MIGRAPHX_THROW("PARSE_GRID_SAMPLE: only 4-D inputs are supported");
        }

        return (mode == "nearest")
                   ? nearest_sampler(x, grid, align_corners, std::move(padding_mode)).sample(info)
                   : linear_sampler(x, grid, align_corners, std::move(padding_mode)).sample(info);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
