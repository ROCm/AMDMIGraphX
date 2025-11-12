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

#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(gridsample_channel_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // gridsample constructor
    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 4}});
    auto grid =
        mm->add_parameter("grid", migraphx::shape{migraphx::shape::float_type, {1, 6, 6, 2}});

    auto m_zero_l =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {0.0f}});
    auto m_one_l =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1.0f}});
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {2}});
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {-0.5f}});
    auto m_width_max_l =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {3}});
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {4}});
    auto m_height_max_l =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {3}});
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {4}});

    auto x_coords = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), grid);
    auto y_coords = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}), grid);

    x_coords = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), x_coords);
    y_coords = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), y_coords);

    // unnorm x (align corners)
    auto m_unnorm_x  = add_common_op(*mm, migraphx::make_op("add"), {x_coords, m_one_l});
    auto mul_const_x = mm->add_literal(
        migraphx::literal{migraphx::shape{x_coords->get_shape().type()}, {(4.f - 1) / 2}});
    m_unnorm_x = add_common_op(*mm, migraphx::make_op("mul"), {m_unnorm_x, mul_const_x});

    // unnorm y (align corners)
    auto m_unnorm_y  = add_common_op(*mm, migraphx::make_op("add"), {y_coords, m_one_l});
    auto mul_const_y = mm->add_literal(
        migraphx::literal{migraphx::shape{y_coords->get_shape().type()}, {(4.f - 1) / 2}});
    m_unnorm_y = add_common_op(*mm, migraphx::make_op("mul"), {m_unnorm_y, mul_const_y});

    // border padding
    m_unnorm_x =
        add_common_op(*mm, migraphx::make_op("clip"), {m_unnorm_x, m_zero_l, m_width_max_l});
    m_unnorm_y =
        add_common_op(*mm, migraphx::make_op("clip"), {m_unnorm_y, m_zero_l, m_height_max_l});

    // linear sampler
    auto m_floor_x = add_common_op(*mm, migraphx::make_op("floor"), {m_unnorm_x});
    auto m_floor_y = add_common_op(*mm, migraphx::make_op("floor"), {m_unnorm_y});
    auto m_ceil_x  = add_common_op(*mm, migraphx::make_op("add"), {m_floor_x, m_one_l});
    auto m_ceil_y  = add_common_op(*mm, migraphx::make_op("add"), {m_floor_y, m_one_l});

    auto fract_x           = add_common_op(*mm, migraphx::make_op("sub"), {m_unnorm_x, m_floor_x});
    auto fract_y           = add_common_op(*mm, migraphx::make_op("sub"), {m_unnorm_y, m_floor_y});
    auto one_minus_fract_x = add_common_op(*mm, migraphx::make_op("sub"), {m_one_l, fract_x});
    auto one_minus_fract_y = add_common_op(*mm, migraphx::make_op("sub"), {m_one_l, fract_y});

    std::array<migraphx::instruction_ref, 4> m_corner_weights;
    m_corner_weights[0] =
        add_common_op(*mm, migraphx::make_op("mul"), {one_minus_fract_y, one_minus_fract_x});
    m_corner_weights[1] =
        add_common_op(*mm, migraphx::make_op("mul"), {one_minus_fract_y, fract_x});
    m_corner_weights[2] =
        add_common_op(*mm, migraphx::make_op("mul"), {fract_y, one_minus_fract_x});
    m_corner_weights[3] = add_common_op(*mm, migraphx::make_op("mul"), {fract_y, fract_x});

    std::vector<float> xy_indices_data(108 * 3);
    std::vector<float> weight_indices_data(108 * 3);
    std::vector<float> nc_values_data(108 * 2);
    auto xy_indices_t = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {108, 3}}, xy_indices_data});
    auto weight_index_t = mm->add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::float_type, {108, 3}}, weight_indices_data});
    auto nc             = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {108, 2}}, nc_values_data});

    auto y0_samples = mm->add_instruction(migraphx::make_op("gathernd"), m_floor_y, xy_indices_t);
    auto x0_samples = mm->add_instruction(migraphx::make_op("gathernd"), m_floor_x, xy_indices_t);
    auto y1_samples = mm->add_instruction(migraphx::make_op("gathernd"), m_ceil_y, xy_indices_t);
    auto x1_samples = mm->add_instruction(migraphx::make_op("gathernd"), m_ceil_x, xy_indices_t);

    auto validate_samples = [&](auto& samples, auto& max) {
        auto clip       = add_common_op(*mm, migraphx::make_op("clip"), {samples, m_zero_l, max});
        auto validation = add_common_op(*mm, migraphx::make_op("equal"), {samples, clip});
        samples         = clip;
        return validation;
    };

    auto y0_validation = validate_samples(y0_samples, m_height_max_l);
    auto x0_validation = validate_samples(x0_samples, m_width_max_l);
    auto y1_validation = validate_samples(y1_samples, m_height_max_l);
    auto x1_validation = validate_samples(x1_samples, m_width_max_l);

    y0_samples = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {y0_samples->get_shape().elements(), 1}}}),
        y0_samples);
    x0_samples = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {x0_samples->get_shape().elements(), 1}}}),
        x0_samples);
    y1_samples = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {y1_samples->get_shape().elements(), 1}}}),
        y1_samples);
    x1_samples = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {x1_samples->get_shape().elements(), 1}}}),
        x1_samples);

    auto make_corner_indices = [&](auto& x, auto& y) {
        auto hw = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), y, x);
        return mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), nc, hw);
    };

    std::array<migraphx::instruction_ref, 4> corner_indices{
        make_corner_indices(x0_samples, y0_samples),
        make_corner_indices(x1_samples, y0_samples),
        make_corner_indices(x0_samples, y1_samples),
        make_corner_indices(x1_samples, y1_samples)};

    std::array<migraphx::instruction_ref, 4> corner_validations{
        add_common_op(*mm, migraphx::make_op("logical_and"), {x0_validation, y0_validation}),
        add_common_op(*mm, migraphx::make_op("logical_and"), {x1_validation, y0_validation}),
        add_common_op(*mm, migraphx::make_op("logical_and"), {x0_validation, y1_validation}),
        add_common_op(*mm, migraphx::make_op("logical_and"), {x1_validation, y1_validation})};

    std::array<migraphx::instruction_ref, 4> corner_samples;
    std::transform(
        corner_indices.begin(),
        corner_indices.end(),
        corner_validations.begin(),
        corner_samples.begin(),
        [&](const auto& indices, const auto& validations) {
            auto samples = mm->add_instruction(migraphx::make_op("gathernd"), x, indices);
            return add_common_op(*mm, migraphx::make_op("where"), {validations, samples, m_zero_l});
        });

    std::transform(corner_samples.begin(),
                   corner_samples.end(),
                   m_corner_weights.begin(),
                   corner_samples.begin(),
                   [&](const auto& samples, const auto& weights) {
                       auto weights_t = mm->add_instruction(
                           migraphx::make_op("gathernd"), weights, weight_index_t);
                       return mm->add_instruction(migraphx::make_op("mul"), samples, weights_t);
                   });

    auto samples = std::accumulate(
        std::next(corner_samples.begin()),
        corner_samples.end(),
        corner_samples.front(),
        [&](auto acc, auto s) { return mm->add_instruction(migraphx::make_op("add"), acc, s); });

    samples = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 6, 6, 3}}}), samples);
    samples = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}),
                                  samples);
    samples = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), samples);

    auto prog = optimize_onnx("gridsample_channel_test.onnx");
    EXPECT(p == prog);
}
