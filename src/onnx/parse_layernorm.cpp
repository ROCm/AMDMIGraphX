/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_layernorm : op_parser<parse_layernorm>
{
    std::vector<op_desc> operators() const { return {{"LayerNormalization"}}; }

    static int64_t handle_axis(const onnx_parser& parser, const onnx_parser::node_info& info)
    {
        int64_t axis = -1;
        if(contains(info.attributes, "axis"))
        {
            axis = parser.parse_value(info.attributes.at("axis")).at<int64_t>();
        }
        return axis;
    }

    static float handle_epsilon(const onnx_parser& parser, const onnx_parser::node_info& info)
    {
        float epsilon = 1e-5f;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }
        return epsilon;
    }

    static bool handle_stash_type(const onnx_parser& parser, const onnx_parser::node_info& info)
    {
        bool stash_type = true;
        if(contains(info.attributes, "stash_type"))
        {
            stash_type = (1 == parser.parse_value(info.attributes.at("stash_type")).at<int64_t>());
        }
        return stash_type;
    }

    static void is_type_valid(const migraphx::shape::type_t& dtype, const std::string& var_name)
    {
        std::set<migraphx::shape::type_t> valid_types = {
            migraphx::shape::float_type, migraphx::shape::bf16_type, migraphx::shape::half_type};

        if(not(contains(valid_types, dtype)))
        {
            MIGRAPHX_THROW("PARSE_LAYERNORM: Invalid type for " + var_name);
        }
    }

    static void check_x_input(const instruction_ref& x, const int64_t& axis)
    {
        auto x_shape   = x->get_shape();
        auto x_dtype   = x_shape.type();
        int64_t x_rank = x_shape.ndim();
        is_type_valid(x_dtype, "input");

        if(x_rank < 2)
        {
            MIGRAPHX_THROW("PARSE_LAYERNORM: invalid ndims=" + std::to_string(x_rank) +
                           ", must be at least 2");
        }

        // If rank(X) is r, axis' allowed range is [-r, r)
        if(axis < -x_rank or axis >= x_rank)
        {
            MIGRAPHX_THROW("PARSE_LAYERNORM: invalid axis");
        }
    }

    static std::tuple<instruction_ref, instruction_ref, instruction_ref>
    stage_one_calculation(const onnx_parser::node_info& info,
                          const instruction_ref& input,
                          const float& epsilon,
                          const int64_t& axis,
                          const int64_t& kdims,
                          bool stash_type)
    {
        // y = (x - mean) * rsqrt(variance + epsilon) * scale + bias
        // mean = reduce_mean({D1, D2, ... Dk}, x)
        // variance = reduce_mean({D1, D2, ... Dk}, (x - mean)^2)

        std::vector<int64_t> axes(kdims);
        std::iota(axes.begin(), axes.end(), axis);
        auto x_shape = input->get_shape();
        auto x_dtype = x_shape.type();

        auto x = input;
        if(stash_type and x_dtype != migraphx::shape::float_type)
        {
            x = info.add_instruction(
                make_op("convert", {{"target_type", migraphx::shape::float_type}}), input);
        }

        auto mean          = info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), x);
        auto x_sub_mean    = info.add_common_op("sub", x, mean);
        auto x_sqdiff_mean = info.add_common_op("sqdiff", x, mean);
        auto variance =
            info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), x_sqdiff_mean);
        auto epsilon_val =
            (x_dtype == migraphx::shape::half_type and std::abs(epsilon) < 1e-7) ? 1e-7 : epsilon;
        auto eps     = info.add_literal(migraphx::literal{migraphx::shape{x_dtype}, {epsilon_val}});
        auto var_eps = info.add_common_op("add", variance, eps);
        auto rsqrt   = info.add_instruction(make_op("rsqrt"), var_eps);
        auto result  = info.add_common_op("mul", x_sub_mean, rsqrt);

        if(stash_type and x_dtype != migraphx::shape::float_type)
        {
            result = info.add_instruction(make_op("convert", {{"target_type", x_dtype}}), result);
        }

        return {result, mean, rsqrt};
    }

    static instruction_ref stage_two_calculation(const onnx_parser::node_info& info,
                                                 const instruction_ref& x,
                                                 const instruction_ref& scale,
                                                 const instruction_ref& bias,
                                                 const instruction_ref& result,
                                                 const int64_t& kdims,
                                                 bool skip_bias)
    {
        auto x_shape                = x->get_shape();
        auto x_rank                 = x_shape.ndim();
        auto skipped_axes           = x_rank - kdims;
        instruction_ref scale_bcast = scale;
        instruction_ref bias_bcast  = bias;
        if(skipped_axes > 0)
        {
            auto x_dims = x_shape.lens();
            if(scale->get_shape().ndim() == 1)
            {
                scale_bcast = info.add_instruction(
                    make_op("broadcast", {{"axis", skipped_axes}, {"out_lens", x_dims}}), scale);
            }

            if(not skip_bias)
            {
                if(bias->get_shape().ndim() == 1)
                {
                    bias_bcast = info.add_instruction(
                        make_op("broadcast", {{"axis", skipped_axes}, {"out_lens", x_dims}}), bias);
                }
            }
        }
        auto scaled = info.add_common_op("mul", result, scale_bcast);
        return skip_bias ? scaled : info.add_common_op("add", scaled, bias_bcast);
    }

    std::tuple<instruction_ref, instruction_ref, instruction_ref, bool>
    handle_inputs(std::vector<instruction_ref>& args, const int64_t& axis) const
    {
        if(args.size() < 2 or args.size() > 3)
        {
            MIGRAPHX_THROW("PARSE_LAYERNORM: invalid input count");
        }
        auto x = args.at(0);
        check_x_input(x, axis);

        auto scale = args.at(1);
        is_type_valid(scale->get_shape().type(), "scale");

        bool skip_bias = args.size() == 2;
        instruction_ref bias;
        if(not skip_bias)
        {
            bias = args.at(2);
            is_type_valid(bias->get_shape().type(), "bias");
        }
        return {x, scale, bias, skip_bias};
    }

    std::tuple<int64_t, float, bool> handle_attributes(const onnx_parser& parser,
                                                       const onnx_parser::node_info& info) const
    {
        auto axis       = handle_axis(parser, info);
        auto epsilon    = handle_epsilon(parser, info);
        auto stash_type = handle_stash_type(parser, info);

        return {axis, epsilon, stash_type};
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       std::vector<instruction_ref> args) const
    {
        auto [axis, epsilon, stash_type] = handle_attributes(parser, info);

        auto [x, scale, bias, skip_bias] = handle_inputs(args, axis);

        auto x_rank = x->get_shape().ndim();
        // axis can be negative
        axis       = axis < 0 ? axis + x_rank : axis;
        auto kdims = x_rank - axis;

        auto [result, mean, rsqrt] =
            stage_one_calculation(info, x, epsilon, axis, kdims, stash_type);
        auto y = stage_two_calculation(info, x, scale, bias, result, kdims, skip_bias);

        return {y, mean, rsqrt};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
