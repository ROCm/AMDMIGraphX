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
#include <migraphx/dfor.hpp>

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
    migraphx::shape m_nc_shape;

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
                 std::string&& padding,
                 const onnx_parser::node_info& info)
        : m_padding(std::move(padding)), m_align_corners(align), m_input(input), m_grid(grid)
    {
        auto i_lens    = input->get_shape().lens();
        m_batch        = i_lens.at(0);
        m_channel      = i_lens.at(1);
        m_in_height    = i_lens.at(2);
        m_in_width     = i_lens.at(3);
        auto g_lens    = grid->get_shape().lens();
        m_out_height   = g_lens.at(1);
        m_out_width    = g_lens.at(2);
        auto type      = m_grid->get_shape().type();
        m_nc_shape     = migraphx::shape{type, {1, 2}};
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

    instruction_ref reflect_coordinates(const onnx_parser::node_info& info,
                                        instruction_ref coords,
                                        instruction_ref size,
                                        instruction_ref corner_start) const
    {
        auto index_align_corner = info.add_common_op("sub", corner_start, coords);
        index_align_corner      = info.add_common_op("abs", index_align_corner);
        auto size_times         = info.add_common_op("floor", index_align_corner);
        size_times              = info.add_common_op("div", size_times, size);
        size_times              = info.add_common_op("floor", size_times);
        auto cond               = info.add_common_op("mod", size_times, m_two_l);
        cond                    = info.add_common_op("equal", cond, m_zero_l);
        auto extra              = info.add_common_op("mul", size_times, size);
        extra                   = info.add_common_op("sub", index_align_corner, extra);
        auto cond_true          = info.add_common_op("add", extra, corner_start);
        auto cond_false         = info.add_common_op("sub", size, extra);
        cond_false              = info.add_common_op("add", cond_false, corner_start);
        return info.add_common_op("where", cond, cond_true, cond_false);
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

    static instruction_ref concat_on_first_dim(const onnx_parser::node_info& info,
                                               std::vector<instruction_ref> instructions)
    {
        return std::accumulate(
            std::next(instructions.begin()),
            instructions.end(),
            instructions.front(),
            [&info](auto& ret, auto& ins) {
                return info.add_instruction(make_op("concat", {{"axis", 0}}), ret, ins);
            });
    }

    inline bool has_border_padding() const { return m_padding == "border"; }
};

struct nearest_sampler : grid_sampler
{
    instruction_ref m_round_x;
    instruction_ref m_round_y;

    nearest_sampler(const instruction_ref& input,
                    const instruction_ref& grid,
                    bool align,
                    std::string&& padding,
                    const onnx_parser::node_info& info)
        : grid_sampler(input, grid, align, std::move(padding), info),
          m_round_x(info.add_common_op("nearbyint", m_unnorm_x)),
          m_round_y(info.add_common_op("nearbyint", m_unnorm_y))
    {
    }

    instruction_ref sample(const onnx_parser::node_info& info)
    {
        std::vector<instruction_ref> hw_indices;
        std::vector<instruction_ref> nc_values;
        const static auto nhw_shape = migraphx::shape{migraphx::shape::int64_type, {1, 3}};
        bool validate               = not has_border_padding();
        dfor(m_batch, m_out_height, m_out_width)([&](auto n, auto h, auto w) {
            auto nhw = info.add_literal(migraphx::literal{nhw_shape, {n, h, w}});
            for(size_t c = 0; c < m_channel; c++)
            {
                hw_indices.push_back(nhw);
                nc_values.push_back(info.add_literal(migraphx::literal{m_nc_shape, {n, c}}));
            }
        });

        auto hw_indices_t = concat_on_first_dim(info, hw_indices);
        auto h_samples    = info.add_instruction(make_op("gathernd"), m_round_y, hw_indices_t);
        auto w_samples    = info.add_instruction(make_op("gathernd"), m_round_x, hw_indices_t);

        instruction_ref validation;
        if(validate)
        {
            auto h_clip  = info.add_common_op("clip", h_samples, m_zero_l, m_height_max_l);
            auto w_clip  = info.add_common_op("clip", w_samples, m_zero_l, m_width_max_l);
            auto h_valid = info.add_common_op("equal", h_samples, h_clip);
            auto w_valid = info.add_common_op("equal", w_samples, w_clip);
            validation   = info.add_common_op("logical_and", h_valid, w_valid);
            h_samples    = h_clip;
            w_samples    = w_clip;
        }

        auto nc   = concat_on_first_dim(info, nc_values);
        h_samples = info.add_instruction(
            make_op("reshape", {{"dims", {h_samples->get_shape().elements(), 1}}}), h_samples);

        w_samples = info.add_instruction(
            make_op("reshape", {{"dims", {w_samples->get_shape().elements(), 1}}}), w_samples);

        auto indices_t =
            info.add_instruction(make_op("concat", {{"axis", 1}}), h_samples, w_samples);
        indices_t = info.add_instruction(make_op("concat", {{"axis", 1}}), nc, indices_t);

        auto samples = info.add_instruction(make_op("gathernd"), m_input, indices_t);
        if(validate)
        {
            samples = info.add_common_op("where", validation, samples, m_zero_l);
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
                   std::string&& padding,
                   const onnx_parser::node_info& info)
        : grid_sampler(input, grid, align, std::move(padding), info),
          m_floor_x(info.add_common_op("floor", m_unnorm_x)),
          m_floor_y(info.add_common_op("floor", m_unnorm_y)),
          m_ceil_x(info.add_common_op("add", m_floor_x, m_one_l)),
          m_ceil_y(info.add_common_op("add", m_floor_y, m_one_l))
    {
        auto fract_x           = info.add_common_op("sub", m_unnorm_x, m_floor_x);
        auto fract_y           = info.add_common_op("sub", m_unnorm_y, m_floor_y);
        auto one_minus_fract_x = info.add_common_op("sub", m_one_l, fract_x);
        auto one_minus_fract_y = info.add_common_op("sub", m_one_l, fract_y);
        m_corner_weights[0]    = info.add_common_op("mul", one_minus_fract_y, one_minus_fract_x);
        m_corner_weights[1]    = info.add_common_op("mul", one_minus_fract_y, fract_x);
        m_corner_weights[2]    = info.add_common_op("mul", fract_y, one_minus_fract_x);
        m_corner_weights[3]    = info.add_common_op("mul", fract_y, fract_x);
    }

    instruction_ref sample(const onnx_parser::node_info& info)
    {
        std::vector<instruction_ref> weight_indices;
        std::vector<instruction_ref> xy_indices;
        std::vector<instruction_ref> nc_values;

        const static auto nhw_shape = migraphx::shape{migraphx::shape::int64_type, {1, 3}};
        dfor(m_batch, m_out_height, m_out_width)([&](auto n, auto h, auto w) {
            auto nhw = info.add_literal(migraphx::literal{nhw_shape, {n, h, w}});
            weight_indices.push_back(nhw);
            for(size_t c = 0; c < m_channel; c++)
            {
                xy_indices.push_back(nhw);
                nc_values.push_back(info.add_literal(migraphx::literal{m_nc_shape, {n, c}}));
            }
        });

        auto xy_indices_t = concat_on_first_dim(info, xy_indices);
        auto y0_samples   = info.add_instruction(make_op("gathernd"), m_floor_y, xy_indices_t);
        auto x0_samples   = info.add_instruction(make_op("gathernd"), m_floor_x, xy_indices_t);
        auto y1_samples   = info.add_instruction(make_op("gathernd"), m_ceil_y, xy_indices_t);
        auto x1_samples   = info.add_instruction(make_op("gathernd"), m_ceil_x, xy_indices_t);

        auto validate_samples = [&](auto& samples, auto& max) {
            auto clip       = info.add_common_op("clip", samples, m_zero_l, max);
            auto validation = info.add_common_op("equal", samples, clip);
            samples         = clip;
            return validation;
        };

        auto y0_validation = validate_samples(y0_samples, m_height_max_l);
        auto x0_validation = validate_samples(x0_samples, m_width_max_l);
        auto y1_validation = validate_samples(y1_samples, m_height_max_l);
        auto x1_validation = validate_samples(x1_samples, m_width_max_l);

        y0_samples = info.add_instruction(
            make_op("reshape", {{"dims", {y0_samples->get_shape().elements(), 1}}}), y0_samples);
        x0_samples = info.add_instruction(
            make_op("reshape", {{"dims", {x0_samples->get_shape().elements(), 1}}}), x0_samples);
        y1_samples = info.add_instruction(
            make_op("reshape", {{"dims", {y1_samples->get_shape().elements(), 1}}}), y1_samples);
        x1_samples = info.add_instruction(
            make_op("reshape", {{"dims", {x1_samples->get_shape().elements(), 1}}}), x1_samples);

        auto nc = concat_on_first_dim(info, nc_values);

        auto make_corner_indices = [&](auto& x, auto& y) {
            auto hw = info.add_instruction(make_op("concat", {{"axis", 1}}), y, x);
            return info.add_instruction(make_op("concat", {{"axis", 1}}), nc, hw);
        };
        std::array<instruction_ref, 4> corner_indices{make_corner_indices(x0_samples, y0_samples),
                                                      make_corner_indices(x1_samples, y0_samples),
                                                      make_corner_indices(x0_samples, y1_samples),
                                                      make_corner_indices(x1_samples, y1_samples)};

        std::array<instruction_ref, 4> corner_validations{
            info.add_common_op("logical_and", x0_validation, y0_validation),
            info.add_common_op("logical_and", x1_validation, y0_validation),
            info.add_common_op("logical_and", x0_validation, y1_validation),
            info.add_common_op("logical_and", x1_validation, y1_validation)};

        std::array<instruction_ref, 4> corner_samples;
        auto weight_index_t = concat_on_first_dim(info, weight_indices);
        weight_index_t      = info.add_instruction(
            make_op("reshape", {{"dims", {weight_indices.size(), 3}}}), weight_index_t);

        std::transform(corner_indices.begin(),
                       corner_indices.end(),
                       corner_validations.begin(),
                       corner_samples.begin(),
                       [&](const auto& indices, const auto& validations) {
                           auto samples =
                               info.add_instruction(make_op("gathernd"), m_input, indices);
                           return info.add_common_op("where", validations, samples, m_zero_l);
                       });

        std::transform(corner_samples.begin(),
                       corner_samples.end(),
                       m_corner_weights.begin(),
                       corner_samples.begin(),
                       [&](const auto& samples, const auto& weights) {
                           auto weights_t =
                               info.add_instruction(make_op("gathernd"), weights, weight_index_t);
                           return info.add_instruction(make_op("mul"), samples, weights_t);
                       });

        auto samples = std::accumulate(
            std::next(corner_samples.begin()),
            corner_samples.end(),
            corner_samples.front(),
            [&](auto acc, auto s) { return info.add_instruction(make_op("add"), acc, s); });

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
                   ? nearest_sampler(x, grid, align_corners, std::move(padding_mode), info)
                         .sample(info)
                   : linear_sampler(x, grid, align_corners, std::move(padding_mode), info)
                         .sample(info);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
