#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

float sigmoid(float x) { return 1 / (1 + expf(-x)); }

float elu(float a, float x) { return x > 0 ? x : a * std::expm1(x); }

TEST_CASE(slice_test)
{
    {
        migraphx::program p;
        std::vector<int> data(2 * 2 * 3);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
        auto l0 = p.add_literal(migraphx::literal{s, data});
        p.add_instruction(migraphx::op::slice{{2}, {1}, {3}}, l0);
        migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
        EXPECT(p.get_shape() == s2);
        p.compile(migraphx::cpu::target{});
        migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
        auto result           = p.eval({});
        std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
        std::vector<int> results_vector(2 * 2 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(result.get_shape() == sresult);
    }
    {
        migraphx::program p;
        std::vector<int> data(2 * 2 * 3);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
        auto l0 = p.add_literal(migraphx::literal{s, data});
        p.add_instruction(migraphx::op::slice{{0, 1, 2}, {0, 0, 0}, {2, 2, 2}}, l0);
        migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
        EXPECT(p.get_shape() == s2);
        p.compile(migraphx::cpu::target{});
        migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
        auto result           = p.eval({});
        std::vector<int> gold = {0, 1, 3, 4, 6, 7, 9, 10};
        std::vector<int> results_vector(2 * 2 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(result.get_shape() == sresult);
    }
}

TEST_CASE(concat_test)
{
    {
        migraphx::program p;
        std::size_t axis       = 1;
        std::vector<int> data0 = {0, 1, 5, 6};
        std::vector<int> data1 = {2, 3, 4, 7, 8, 9};
        std::vector<int> data2 = {10, 20};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::int32_type, {2, 1}};
        auto l0 = p.add_literal(migraphx::literal{s0, data0});
        auto l1 = p.add_literal(migraphx::literal{s1, data1});
        auto l2 = p.add_literal(migraphx::literal{s2, data2});
        p.add_instruction(migraphx::op::concat{axis}, l0, l1, l2);
        p.compile(migraphx::cpu::target{});
        auto result           = p.eval({});
        std::vector<int> gold = {0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9, 20};
        std::vector<int> results_vector(2 * 6);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(migraphx::verify_range(result.get_shape().lens(), std::vector<std::size_t>({2, 6})));
        EXPECT(
            migraphx::verify_range(result.get_shape().strides(), std::vector<std::size_t>({6, 1})));
    }
    {
        migraphx::program p;
        std::size_t axis       = 0;
        std::vector<int> data0 = {0, 1, 2, 3};
        std::vector<int> data1 = {4, 5, 6, 7, 8, 9};
        std::vector<int> data2 = {10, 11};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::int32_type, {1, 2}};
        auto l0 = p.add_literal(migraphx::literal{s0, data0});
        auto l1 = p.add_literal(migraphx::literal{s1, data1});
        auto l2 = p.add_literal(migraphx::literal{s2, data2});
        p.add_instruction(migraphx::op::concat{axis}, l0, l1, l2);
        p.compile(migraphx::cpu::target{});
        auto result           = p.eval({});
        std::vector<int> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<int> results_vector(6 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(migraphx::verify_range(result.get_shape().lens(), std::vector<std::size_t>({6, 2})));
        EXPECT(
            migraphx::verify_range(result.get_shape().strides(), std::vector<std::size_t>({2, 1})));
    }
}

TEST_CASE(gather_test)
{
    {
        migraphx::program p;

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = p.add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{0, 2};
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = 0;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }

    {
        migraphx::program p;

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = p.add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{0, 2};
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = 1;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }

    {
        migraphx::program p;

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = p.add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{0, 2};
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }
}

TEST_CASE(squeeze_test)
{
    {
        migraphx::program p;
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
        auto l0 = p.add_literal(migraphx::literal{s1, data});
        p.add_instruction(migraphx::op::squeeze{{1}}, l0);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        EXPECT(result.get_shape() == s2);
    }
    {
        migraphx::program p;
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
        auto l0 = p.add_literal(migraphx::literal{s1, data});
        p.add_instruction(migraphx::op::squeeze{{3}}, l0);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        EXPECT(result.get_shape() == s2);
    }
    {
        migraphx::program p;
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 3, 3}};
        auto l0 = p.add_literal(migraphx::literal{s1, data});
        p.add_instruction(migraphx::op::squeeze{}, l0);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        EXPECT(result.get_shape() == s2);
    }
}

TEST_CASE(unsqueeze_test)
{
    {
        migraphx::program p;
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
        auto l0 = p.add_literal(migraphx::literal{s1, data});
        p.add_instruction(migraphx::op::unsqueeze{{1}}, l0);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        EXPECT(result.get_shape() == s2);
    }
    {
        migraphx::program p;
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
        auto l0 = p.add_literal(migraphx::literal{s1, data});
        p.add_instruction(migraphx::op::unsqueeze{{2}}, l0);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        EXPECT(result.get_shape() == s2);
    }
}

TEST_CASE(globalavgpool_test)
{
    migraphx::program p;
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op    = migraphx::op::pooling{"average"};
    auto lens  = s.lens();
    op.lengths = {lens[2], lens[3]};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(op, l0);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.575, 0.375};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(globalmaxpool_test)
{
    migraphx::program p;
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op    = migraphx::op::pooling{"max"};
    auto lens  = s.lens();
    op.lengths = {lens[2], lens[3]};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(op, l0);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.9, 0.7};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(im2col_3x3_no_pad_identity_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {3, 3};
    std::array<std::size_t, 2> padding{{0, 0}};
    std::array<std::size_t, 2> stride{{1, 1}};
    std::array<std::size_t, 2> dilation{{1, 1}};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = p.add_literal(migraphx::literal{s_image, input});
    auto l_weights = p.add_literal(migraphx::literal{s_weights, weights});
    p.add_instruction(migraphx::op::im2col{padding, stride, dilation}, l_image, l_weights);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, input));
}

TEST_CASE(im2col_3x3_no_pad_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {4, 4};
    std::array<std::size_t, 2> padding{{0, 0}};
    std::array<std::size_t, 2> stride{{1, 1}};
    std::array<std::size_t, 2> dilation{{1, 1}};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = p.add_literal(migraphx::literal{s_image, input});
    auto l_weights = p.add_literal(migraphx::literal{s_weights, weights});
    p.add_instruction(migraphx::op::im2col{padding, stride, dilation}, l_image, l_weights);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<int> correct = {0, 1, 2, 4, 5, 6,  8,  9,  10, 1, 2, 3, 5, 6,  7,  9,  10, 11,
                                4, 5, 6, 8, 9, 10, 12, 13, 14, 5, 6, 7, 9, 10, 11, 13, 14, 15};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, correct));
}

TEST_CASE(im2col_3x3_stride_2_no_pad_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {6, 6};
    std::array<std::size_t, 2> padding{{0, 0}};
    std::array<std::size_t, 2> stride{{2, 2}};
    std::array<std::size_t, 2> dilation{{1, 1}};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = p.add_literal(migraphx::literal{s_image, input});
    auto l_weights = p.add_literal(migraphx::literal{s_weights, weights});
    p.add_instruction(migraphx::op::im2col{padding, stride, dilation}, l_image, l_weights);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<int> correct = {0,  1,  2,  6,  7,  8,  12, 13, 14, 2,  3,  4,
                                8,  9,  10, 14, 15, 16, 12, 13, 14, 18, 19, 20,
                                24, 25, 26, 14, 15, 16, 20, 21, 22, 26, 27, 28};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, correct));
}

TEST_CASE(im2col_3x3_with_padding_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {2, 2};
    std::array<std::size_t, 2> padding{{1, 1}};
    std::array<std::size_t, 2> stride{{1, 1}};
    std::array<std::size_t, 2> dilation{{1, 1}};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = p.add_literal(migraphx::literal{s_image, input});
    auto l_weights = p.add_literal(migraphx::literal{s_weights, weights});
    p.add_instruction(migraphx::op::im2col{padding, stride, dilation}, l_image, l_weights);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<int> correct = {0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0,
                                0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, correct));
}

TEST_CASE(batch_norm_inference_test)
{
    migraphx::program p;
    const size_t width       = 2;
    const size_t height      = 2;
    const size_t channels    = 4;
    const size_t batches     = 2;
    const float x_val        = 8.0;
    const float mean_val     = 2.0;
    const float variance_val = 4.0;
    const float scale_val    = 2.0f;
    const float bias_val     = 1.0f;
    const float output_val = scale_val * (x_val - mean_val) / (std::sqrt(variance_val)) + bias_val;

    migraphx::shape s{migraphx::shape::float_type, {batches, channels, height, width}};
    migraphx::shape vars{migraphx::shape::float_type, {channels}};
    std::vector<float> x_data(width * height * channels * batches);
    std::vector<float> scale_data(channels);
    std::vector<float> bias_data(channels);
    std::vector<float> mean_data(channels);
    std::vector<float> variance_data(channels);

    std::fill(x_data.begin(), x_data.end(), x_val);
    std::fill(mean_data.begin(), mean_data.end(), mean_val);
    std::fill(variance_data.begin(), variance_data.end(), variance_val);
    std::fill(scale_data.begin(), scale_data.end(), scale_val);
    std::fill(bias_data.begin(), bias_data.end(), bias_val);

    auto x        = p.add_literal(migraphx::literal{s, x_data});
    auto scale    = p.add_literal(migraphx::literal{vars, scale_data});
    auto bias     = p.add_literal(migraphx::literal{vars, bias_data});
    auto mean     = p.add_literal(migraphx::literal{vars, mean_data});
    auto variance = p.add_literal(migraphx::literal{vars, variance_data});

    p.add_instruction(migraphx::op::batch_norm_inference{}, x, scale, bias, mean, variance);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> result_vector(width * height * channels * batches);
    std::vector<float> gold(width * height * channels * batches);
    std::fill(gold.begin(), gold.end(), output_val);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(im2col_3x3_with_channels_identity_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {3, 3};
    std::array<std::size_t, 2> padding{{0, 0}};
    std::array<std::size_t, 2> stride{{1, 1}};
    std::array<std::size_t, 2> dilation{{1, 1}};
    std::size_t channels = 2;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = p.add_literal(migraphx::literal{s_image, input});
    auto l_weights = p.add_literal(migraphx::literal{s_weights, weights});
    p.add_instruction(migraphx::op::im2col{padding, stride, dilation}, l_image, l_weights);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, input));
}

TEST_CASE(exp_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {-1, 0, 1}});
    p.add_instruction(migraphx::op::exp{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.36787944f, 1.f, 2.71828183f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(log_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {1, 2, 3}});
    p.add_instruction(migraphx::op::log{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.0f, 0.6931471806f, 1.0986122887f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(sin_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {-1, 0, 1}});
    p.add_instruction(migraphx::op::sin{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.84147098f, 0.f, 0.84147098f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(cos_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {-1, 0, 1}});
    p.add_instruction(migraphx::op::cos{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.54030231f, 1.f, 0.54030231f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(tan_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {-1, 0, 1}});
    p.add_instruction(migraphx::op::tan{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-1.55740772f, 0.0f, 1.55740772f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(asin_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-0.5f, 0.0f, 0.9f};
    auto l = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::asin{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.5235987756f, 0.f, 1.119769515};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(acos_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{-0.8f, 0.0f, 1.0f};
    auto l = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::acos{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {2.4980915448f, 1.5707963268f, 0.0f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(atan_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::double_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {-1, 0, 1}});
    p.add_instruction(migraphx::op::atan{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.7853981634f, 0.0f, 0.7853981634f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(add_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = p.add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = p.add_literal(migraphx::literal{s, {1, 2, 3}});
    p.add_instruction(migraphx::op::add{}, l1, l2);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 2, 4};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(broadcast_test)
{
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    uint64_t axis = 0;
    auto l1       = p.add_literal(migraphx::literal{a_shape, a_data});
    auto l2       = p.add_literal(migraphx::literal{b_shape, b_data});
    p.add_instruction(migraphx::op::broadcast{axis, l1->get_shape()}, l2);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    auto output = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -2);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == -3);
}
TEST_CASE(add_broadcast_test)
{
    {
        migraphx::program p;
        migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3}};
        std::vector<float> a_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        migraphx::shape b_shape{migraphx::shape::float_type, {2, 2}};
        std::vector<float> b_data{0, -1, -2, -3};
        uint64_t axis = 0;
        auto l1       = p.add_literal(migraphx::literal{a_shape, a_data});
        auto l2       = p.add_literal(migraphx::literal{b_shape, b_data});
        auto l3       = p.add_instruction(migraphx::op::broadcast{axis, l1->get_shape()}, l2);
        p.add_instruction(migraphx::op::add{}, l1, l3);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        EXPECT(result.get_shape().packed());
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
    {
        migraphx::program p;
        migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3}};
        std::vector<float> a_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        migraphx::shape b_shape{migraphx::shape::float_type, {2, 2, 1}};
        std::vector<float> b_data{0, -1, -2, -3};
        auto l1 = p.add_literal(migraphx::literal{a_shape, a_data});
        auto l2 = p.add_literal(migraphx::literal{b_shape, b_data});
        auto l3 = p.add_instruction(migraphx::op::multibroadcast{{2, 2, 3}}, l1);
        auto l4 = p.add_instruction(migraphx::op::multibroadcast{{2, 2, 3}}, l2);
        p.add_instruction(migraphx::op::add{}, l3, l4);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        EXPECT(result.get_shape().packed());
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
}

TEST_CASE(sub_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = p.add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = p.add_literal(migraphx::literal{s, {1, 2, 3}});
    p.add_instruction(migraphx::op::sub{}, l1, l2);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2, -2, -2};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(mul_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = p.add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = p.add_literal(migraphx::literal{s, {1, 2, 3}});
    p.add_instruction(migraphx::op::mul{}, l1, l2);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-1, 0, 3};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(div_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = p.add_literal(migraphx::literal{s, {-1.0f, 0.5f, 1.0f}});
    auto l2 = p.add_literal(migraphx::literal{s, {1.0f, 2.0f, 4.0f}});
    p.add_instruction(migraphx::op::div{}, l1, l2);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-1.f, 0.25f, 0.25f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(relu_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {-1.f, 0.f, 1.f}});
    p.add_instruction(migraphx::op::relu{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.f, 0.f, 1.f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(leaky_relu_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {-1.f, 0.f, 1.f}});
    p.add_instruction(migraphx::op::leaky_relu{0.01}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.01f, 0.f, 1.f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(imagescaler_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto img           = p.add_literal(migraphx::literal{s,
                                               {0.2,
                                                0.3,
                                                0.5,
                                                0.4,

                                                0.7,
                                                0.8,
                                                0.1,
                                                0.9,

                                                0.15,
                                                0.25,
                                                0.35,
                                                0.45}});
    auto scale_val     = p.add_literal(2.f);
    auto scaled_tensor = p.add_instruction(migraphx::op::scalar{s}, scale_val);
    auto img_scaled    = p.add_instruction(migraphx::op::mul{}, img, scaled_tensor);
    auto bias_vals     = p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0.01, 0.02, 0.03}});
    auto bias_bcast = p.add_instruction(migraphx::op::broadcast{1, s}, bias_vals);
    p.add_instruction(migraphx::op::add{}, img_scaled, bias_bcast);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.41,
                               0.61,
                               1.01,
                               0.81,

                               1.42,
                               1.62,
                               0.22,
                               1.82,

                               0.33,
                               0.53,
                               0.73,
                               0.93};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(reshape_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), -3);
    {
        migraphx::program p;
        auto l                         = p.add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> new_shape = {8, 3, 1, 1};
        p.add_instruction(migraphx::op::reshape{new_shape}, l);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<float> results_vector(3);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, data));
    }
    {
        migraphx::program p;
        auto l                         = p.add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> new_shape = {1, 3, 4, 2};
        p.add_instruction(migraphx::op::reshape{new_shape}, l);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<float> results_vector(3);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, data));
    }
    {
        migraphx::program p;
        auto l                         = p.add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> new_shape = {1, 3, 4, 2};
        p.add_instruction(migraphx::op::reshape{new_shape}, l);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<float> results_vector(3);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, data));
    }
}

template <class T>
void gemm_test()
{
    migraphx::program p;
    std::vector<T> a     = {-0.00925222, 0.56250403, 0.70107397,  0.75402161,  -0.505885,
                        1.33628943,  -0.11413,   -0.31270559, 1.59336732,  -0.19361027,
                        -0.91620867, 0.40108416, -0.06969921, 0.68483471,  -0.39906632,
                        -1.66423624, 0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<float> b = {6.09568541e-01,
                            -6.10527007e-01,
                            3.66646462e-01,
                            1.18951101e-01,
                            5.58777432e-01,
                            -3.21296298e-01,
                            -5.95997198e-01,
                            -5.01425721e-01,
                            -2.84606807e-01,
                            -5.73673557e-01,
                            -8.99430260e-01,
                            -4.25103093e-01,
                            1.53027987e+00,
                            -3.81407415e-04,
                            -3.29650255e-01};
    std::vector<float> c = {-1.56327541e+00,
                            -7.09570140e-01,
                            -5.37424982e-01,
                            -2.22994831e-01,
                            -2.15586437e+00,
                            2.09177941e-03,
                            -1.47279677e+00,
                            2.02627040e-01,
                            -6.04527691e-01,
                            -1.29885596e+00,
                            2.16294914e+00,
                            -1.48101497e-01};
    migraphx::shape a_shape{migraphx::shape::get_type<T>{}, {4, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::get_type<T>{}, {5, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    p.add_instruction(migraphx::op::dot{}, al, bl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<T> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(c, results_vector));
}
TEST_CASE_REGISTER(gemm_test<float>)
TEST_CASE_REGISTER(gemm_test<double>)

TEST_CASE(maxpool_test)
{
    migraphx::program p;
    std::vector<float> a = {
        -2.1314404,  -1.63041711, 1.54562736,  1.04625261,  -1.42931843, -0.48703974, 0.4065806,
        -0.1524526,  1.30775225,  0.45538983,  -0.06631992, -1.75332725, 1.33493888,  0.47327688,
        0.36873096,  1.18358743,  -0.34640595, 1.22098756,  0.01946825,  -0.20238149, 0.43348005,
        -0.67991608, -0.83041084, 0.93537551,  0.70241445,  -0.5654031,  -1.30899191, -0.26735824,
        -0.52444768, 1.99097753,  1.86504853,  -0.26506025, 0.26236168,  0.43763575,  0.95300823,
        -1.02733946, -0.74655169, -0.5374338,  -0.28901565, -0.59789604, 0.5310151,   0.99125904,
        0.40609556,  -1.57175648, 0.22031412,  1.45862222,  0.53217483,  1.39087725,  1.00170159,
        -0.87175864, -1.7204628,  -1.72008383, -0.38656762, -0.01443311, 1.46645272,  -1.39995027,
        0.22505587,  -0.43461126, -0.05511411, -0.79950953, -0.01439556, 0.08795211,  1.18943918,
        -0.84079367, -1.73383629, -0.55662078, -0.30626822, -0.67339015, 0.44179603,  0.54316711,
        0.40899998,  -0.27831686, -1.11900508, -0.0881724,  0.35483059,  2.36277103,  -0.04765317,
        -0.36865309, 0.73814237,  1.47151589,  1.36546791,  -0.32649881, -1.0517807,  2.24768877,
        0.68883753,  0.58646208,  -0.91017133, -0.50462508, -0.4013325,  -0.72348958, -0.47368807,
        0.35285577,  -1.01817429, -0.5152272,  0.60321307,  0.43521205,  -0.23733577, 0.66427642,
        0.82949388,  0.82443929,  0.71550399,  0.34561086,  0.68570769,  -0.40718508, -1.20350206,
        0.15793853,  -2.31013632, -0.07934658, -0.09348056, 0.36576006,  2.46601582,  0.11090943,
        0.9144392,   0.56759721,  -0.22112127, -0.21955389, 0.72474903,  -1.28448462, 1.53285873,
        0.37437943,  0.31409341,  1.95433736,  0.91620457,  0.86205518,  1.24365854,  0.19248386,
        0.22526583,  0.13462132,  -0.27561715, -2.06446075, -0.02306402, -1.38278747, 1.1411345,
        1.31293464,  -1.86041689, 1.06763375,  -0.26541466, 1.4545635,   1.11430049,  -0.66491818,
        0.87101674,  0.67768967,  -1.02062869, -1.05031872, -2.2764678,  -2.0200038,  0.37592548,
        -0.26701379, -0.83388507, 0.19403623,  1.00968623,  0.11020003,  1.16736257,  -1.1160326,
        0.47346735,  0.6126079,   -0.19135755, 1.33624589,  -0.29802522, -0.57873946, -1.06555879,
        -0.20686582, 1.36892557,  -0.19937795, 0.8649236,   -1.40126073, 1.53441942,  0.34682792,
        -1.31724346, -1.32898355, 2.40126371,  0.07845283,  1.35732043,  -0.63678312, 0.39429256,
        -1.36487007, -0.31026676, -0.44981545, -0.28994772, -0.14657612, -1.75206447, -0.70612341,
        1.20071781,  -1.64647579, -0.7133292,  0.88494766,  0.52119428,  -2.77387547, 2.07681108,
        -0.90133125, 0.2847338,   0.6174528,   -0.20616426, -0.64263535, -1.08496261, 0.54275119,
        -0.88503587, 0.6629802,   1.47319221,  -1.05829155, -0.97027361, -0.93187737, -1.39954746,
        -0.52359426, -0.14743951, 1.51522756,  0.2078452,   -1.28156149, -1.19363916, -0.78680223,
        -0.89094824, 1.30212069,  -0.77974445, -0.58411664, 0.48764706,  -0.67132682};
    std::vector<float> c = {1.33493888, 1.54562736, 1.22098756, 1.33493888, 1.18358743, 1.99097753,
                            1.00170159, 1.45862222, 1.39087725, 1.46645272, 1.18943918, -0.01443311,
                            1.47151589, 2.36277103, 2.24768877, 0.68883753, 0.82949388, 0.71550399,
                            1.95433736, 2.46601582, 1.53285873, 1.95433736, 1.06763375, 1.4545635,
                            1.33624589, 1.16736257, 0.6126079,  1.36892557, 2.40126371, 1.53441942,
                            0.52119428, 2.07681108, 0.88494766, 1.51522756, 0.54275119, 0.6629802};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 6, 6}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    p.add_instruction(migraphx::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{3, 2}}}, al);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    // std::cout << result.get_shape() << std::endl;
    std::vector<float> results_vector(36);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, c));
}

TEST_CASE(softmax_test)
{
    migraphx::program p;
    std::vector<float> a = {
        -5.61869681e-01, 9.07827199e-01,  1.29255986e+00,  3.18533443e-02,  -1.22183852e-03,
        -2.83830553e-01, -1.03245842e+00, -9.28322077e-01, -8.82696748e-01, 1.11327164e-01,
        -9.20038462e-01, 8.47388089e-01,  2.51734018e-01,  1.50563884e+00,  2.23056650e+00,
        -6.17576987e-02, -1.00264274e-01, -6.10369384e-01, 1.17537189e+00,  -2.51560897e-01,
        -8.50333512e-01, -8.03578615e-01, -6.51194930e-01, -2.58137047e-01, 4.65528190e-01,
        3.23284641e-02,  -1.54700470e+00, 1.38096774e+00,  5.39869189e-01,  -7.56884992e-01,
        1.81503093e+00,  -2.11269641e+00, 1.92466557e+00,  1.77230799e+00,  2.21660900e+00,
        1.56777036e+00,  -2.08995026e-03, 3.50566894e-01,  -1.15042710e+00, -1.18577778e+00,
        8.90633047e-01,  -6.63949102e-02, 1.44661188e+00,  1.59215283e+00,  -2.56262213e-01,
        9.39079225e-01,  4.07298543e-02,  3.86590779e-01,  6.09607756e-01,  8.22331488e-01,
        -2.82126725e-01, -9.49052632e-01, -4.24012303e-01, -5.32990396e-01, -3.18386006e+00,
        3.27092171e-01,  -1.33315325e+00, 3.62459183e-01,  3.74710828e-01,  -1.30302286e+00,
        1.79680198e-01,  -4.51832324e-01, 4.34282750e-01,  -7.09520102e-01, 6.20333970e-01,
        -1.28712380e+00, 2.04130828e-01,  -7.70607769e-01, 1.61889160e+00,  -1.50951004e+00,
        -4.10505563e-01, -3.56566496e-02, -1.29747534e+00, -1.49967879e-01, 7.77626812e-01,
        -8.28408226e-02, 2.73412596e-02,  5.79780899e-03,  9.87900198e-02,  -7.95276761e-01,
        -1.38536084e+00, -6.63573861e-01, 3.89783204e-01,  -1.30670881e+00, -7.62425125e-01,
        -4.04883057e-01, 6.24344349e-01,  3.68128955e-01,  -1.01577950e+00, -3.06715906e-01,
        5.67961395e-01,  2.98198581e-01,  -1.63613629e+00, -3.75131965e-01, -6.75393403e-01,
        2.59172034e+00,  6.75538957e-01,  9.07939598e-02,  1.92257717e-01,  -1.21592450e+00,
        -2.73682117e-01, 1.25232983e+00,  -1.39969170e+00, -1.91483587e-01, 2.57732719e-01,
        3.10056299e-01,  1.41833842e+00,  -1.81386679e-01, 3.92868072e-01,  -8.14771175e-01,
        2.02392387e+00,  -9.42091495e-02, -3.77683818e-01, 2.05638766e+00,  2.93796062e-01,
        -6.02131486e-01, 2.70461679e-01,  -8.92358482e-01, 1.04388881e+00,  2.66154885e-01};

    std::vector<float> s = {
        0.30191708, 0.59879845, 0.50029165, 0.24915339, 0.36823985, 0.13190967, 0.0349741,
        0.18750034, 0.21905553, 0.27000085, 0.0547399,  0.56318235, 0.47422904, 0.78964758,
        0.91381913, 0.44601166, 0.47902739, 0.13120073, 0.4449684,  0.18766427, 0.15753111,
        0.07844277, 0.05120674, 0.36648798, 0.14637007, 0.13152322, 0.01560997, 0.29065287,
        0.49196178, 0.10550152, 0.81890774, 0.06369215, 0.62972021, 0.74931765, 0.67285055,
        0.35034987, 0.28612873, 0.31931475, 0.04220394, 0.16093165, 0.22390974, 0.11915915,
        0.3115395,  0.35899726, 0.22190949, 0.57518375, 0.13888834, 0.7753762,  0.4642328,
        0.57055861, 0.21954368, 0.34515455, 0.09486015, 0.40631217, 0.01842281, 0.48770609,
        0.06652815, 0.36023033, 0.42343026, 0.24226256, 0.17348589, 0.44066274, 0.6865865,
        0.17296699, 0.46923906, 0.06921105, 0.3570261,  0.4125829,  0.73165393, 0.15302512,
        0.29499072, 0.33932695, 0.30852377, 0.40762195, 0.40170741, 0.36259529, 0.60848355,
        0.42618036, 0.31721094, 0.02960522, 0.28256637, 0.24389413, 0.2725659,  0.10663581,
        0.27622163, 0.28264219, 0.53652936, 0.09476089, 0.40890986, 0.34848392, 0.32572666,
        0.53076893, 0.11529481, 0.29117745, 0.14625968, 0.8756339,  0.49818122, 0.10656087,
        0.1813329,  0.17664003, 0.21410346, 0.80408043, 0.02315119, 0.27155462, 0.32804728,
        0.13268511, 0.61795473, 0.49703068, 0.41696799, 0.10175809, 0.71028161, 0.29929739,
        0.17377149, 0.76075399, 0.20071237, 0.32632929, 0.36892858, 0.09416146, 0.26656723,
        0.42914796};

    migraphx::shape a_shape{migraphx::shape::float_type, {5, 3, 4, 2}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    p.add_instruction(migraphx::op::softmax{}, al);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(120);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(conv2d_test)
{
    migraphx::program p;
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        2.82721668e-02,  6.44195229e-02,  1.53499246e-02,  1.72468081e-01,  -6.33238107e-02,
        9.49496776e-02,  1.40258059e-01,  -7.92879611e-02, -1.29301161e-01, 3.11307609e-03,
        -1.90624535e-01, 1.13238767e-01,  -2.80647576e-02, 3.12882811e-02,  -3.52091640e-02,
        3.33581865e-02,  6.43158704e-02,  7.40238279e-02,  -1.00106120e-01, -9.56912562e-02,
        1.44342467e-01,  9.40258950e-02,  6.36333972e-02,  1.66158378e-03,  -8.91554281e-02,
        2.58734226e-02,  1.70919895e-02,  1.78214177e-01,  8.84564668e-02,  8.98126513e-02,
        -1.63809001e-01, 1.37802169e-01,  1.66439757e-01,  -1.45631135e-02, 1.88469887e-04,
        4.76950556e-02,  -1.91969007e-01, -1.76233292e-01, -7.70473927e-02, 1.14828631e-01,
        1.76608220e-01,  -1.50728196e-01, 1.99946314e-02,  -5.88052124e-02, 1.31612435e-01,
        1.61106288e-02,  -1.35080189e-01, 1.49512306e-01,  3.86456847e-02,  1.29330024e-01,
        -3.22975963e-02, -5.60784787e-02, -5.41997552e-02, 4.78562862e-02};

    std::vector<float> s = {0.27039781,
                            0.19105849,
                            -0.06339942,
                            -0.65087199,
                            0.40867025,
                            0.05063812,
                            -0.14907975,
                            0.49018705,
                            -0.49197209,
                            0.33236548,
                            -0.39374301,
                            0.16012701,
                            0.06574871,
                            0.71606487,
                            -0.55201721,
                            -0.46427044};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = p.add_literal(migraphx::literal{c_shape, c});

    p.add_instruction(migraphx::op::convolution{}, al, cl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(conv2d_padding_test)
{
    migraphx::program p;
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.16115488, -0.09800646, -0.05412646, 0.10475694,  0.00555485,  -0.12667653, 0.0458357,
        -0.02656217, -0.16338061, 0.15037455,  0.0102711,   0.01303349,  0.05242859,  0.02034754,
        0.04751867,  -0.17038961, -0.1434752,  -0.10770349, 0.05676742,  -0.15838449, 0.10128359,
        -0.18958683, 0.11954515,  0.10758857,  -0.01058291, -0.12797487, 0.08971019,  0.18793164,
        -0.00881396, -0.06588994, -0.13321903, -0.03300409, 0.01439607,  0.07618178,  -0.11556662,
        0.00764295,  0.12956454,  -0.08937147, -0.12763587, 0.04674943,  0.05765297,  0.11336918,
        0.14747436,  -0.06199479, -0.01166052, -0.12432006, -0.04494537, -0.17581205, 0.09475745,
        0.1149437,   -0.1014564,  0.0274073,   -0.01323579, -0.11092556};

    std::vector<float> s = {
        -0.0201216,  0.40407312,  -0.39005592, -0.0631946,  0.37963012,  -0.64611685, 0.1349397,
        -0.54113752, 0.28533003,  0.27667275,  -0.16442731, -0.181494,   0.30564839,  0.58744538,
        0.32015014,  0.24969585,  -0.27367792, -0.53308117, 0.41236052,  0.26136363,  -0.01489828,
        0.57652152,  -0.38506854, 0.119615,    0.0437076,   0.04779706,  0.57887721,  0.23126155,
        0.05695833,  -0.68200272, 0.02063358,  -0.10267162, 0.8062973,   -0.38149622, -0.40134856,
        -0.03353126, 0.38991132,  -0.3478111,  0.03661491,  0.25783631,  0.62772679,  -0.1961118,
        0.76423508,  -0.36241418, -0.20994355, -0.12368261, -0.9406727,  0.02340185,  -0.08793129,
        -0.02471633, -0.58163726, -0.02211772, -0.42014724, 0.77525634,  0.504951,    -0.20537445,
        -0.20369984, -0.83037728, -1.40423918, -0.46160448, -0.22944322, 0.36074194,  0.49579027,
        0.46527559};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = p.add_literal(migraphx::literal{c_shape, c});

    p.add_instruction(migraphx::op::convolution{{{1, 1}}, {{1, 1}}}, al, cl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> results_vector(64);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(conv2d_padding_stride_test)
{
    migraphx::program p;
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.14601797, -0.13000923, 0.06521662,  0.06178288,  -0.11083675, 0.10154136,  0.09990512,
        0.06030385,  -0.11374587, -0.17523311, -0.14344215, 0.17802463,  0.06300922,  -0.15325832,
        0.07066704,  0.05166031,  0.00615084,  -0.02606523, 0.08083995,  -0.17913306, 0.0624622,
        0.0735731,   -0.04198661, -0.0164391,  -0.06374192, 0.16569914,  0.10681538,  0.07370754,
        0.02802075,  0.00282027,  0.15104802,  -0.11084409, -0.00197773, 0.07924436,  0.03528272,
        0.04765259,  -0.15896152, 0.07917164,  0.12125669,  -0.1154705,  -0.11999125, 0.12749968,
        -0.06269585, 0.18658121,  -0.03944227, 0.0111798,   -0.17731084, 0.11789055,  -0.09982193,
        0.08142821,  0.0729029,   0.11303909,  0.12735154,  0.03885292};

    std::vector<float> s = {-0.20817225,
                            0.87965256,
                            0.14958936,
                            -1.24887264,
                            -0.06540672,
                            0.20778663,
                            0.40456355,
                            -0.99900877,
                            0.4917807,
                            0.1994698,
                            0.64205718,
                            0.37798831,
                            -0.25315839,
                            0.44276932,
                            -0.16138598,
                            0.79344082};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = p.add_literal(migraphx::literal{c_shape, c});

    p.add_instruction(migraphx::op::convolution{{{1, 1}}, {{2, 2}}}, al, cl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(transpose_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2, 2, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);

    {
        migraphx::program p;
        auto l                    = p.add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> perm = {0, 3, 1, 2};
        p.add_instruction(migraphx::op::transpose{perm}, l);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});

        result.visit([&](auto output) {
            std::vector<size_t> new_lens = {1, 3, 2, 2};
            EXPECT(bool{output.get_shape().lens() == new_lens});
        });
    }
    {
        migraphx::program p;
        auto l                    = p.add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> perm = {0, 3, 1, 2};
        auto result               = p.add_instruction(migraphx::op::transpose{perm}, l);
        p.add_instruction(migraphx::op::contiguous{}, result);
        p.compile(migraphx::cpu::target{});
        auto result2 = p.eval({});

        std::vector<float> results_vector(12);
        result2.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
}

TEST_CASE(contiguous_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);

    migraphx::program p;
    auto l = p.add_literal(migraphx::literal{a_shape, data});
    p.add_instruction(migraphx::op::contiguous{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<size_t> new_lens    = {1, 3, 2, 2};
    std::vector<size_t> new_strides = {12, 1, 6, 3};
    std::vector<float> gold         = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(identity_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<int> data{1, 2, 3, 4};
    auto l = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::identity{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<int> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::equal(data.begin(), data.end(), results_vector.begin()));
}

TEST_CASE(abs_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l = p.add_literal(migraphx::literal{s, {-1, 2, -3, 4}});
    p.add_instruction(migraphx::op::abs{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 2, 3, 4};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(sigmoid_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l = p.add_literal(migraphx::literal{s, {-1, 2, -3, 4}});
    p.add_instruction(migraphx::op::sigmoid{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{sigmoid(-1), sigmoid(2), sigmoid(-3), sigmoid(4)};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(sinh_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l = p.add_literal(migraphx::literal{s, {-1.0, 2.0, -3.0, 4.0}});
    p.add_instruction(migraphx::op::sinh{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{sinhf(-1), sinhf(2), sinhf(-3), sinhf(4)};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(cosh_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l = p.add_literal(migraphx::literal{s, {-1.0, 2.0, -3.0, 4.0}});
    p.add_instruction(migraphx::op::cosh{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{coshf(-1), coshf(2), coshf(-3), coshf(4)};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(tanh_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l = p.add_literal(migraphx::literal{s, {-1.0, 2.0, -3.0, 4.0}});
    p.add_instruction(migraphx::op::tanh{}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{tanhf(-1), tanhf(2), tanhf(-3), tanhf(4)};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(elu_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l      = p.add_literal(migraphx::literal{s, {-1.0, 2.0, -3.0, 4.0}});
    float alpha = 0.5;
    p.add_instruction(migraphx::op::elu{alpha}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{elu(alpha, -1), elu(alpha, 2), elu(alpha, -3), elu(alpha, 4)};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(max_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l0       = p.add_literal(migraphx::literal{s, {1, 4, 3}});
    auto l1       = p.add_literal(migraphx::literal{s, {2, 8, 6}});
    auto l2       = p.add_literal(migraphx::literal{s, {7, 5, 9}});
    auto curr_max = p.add_instruction(migraphx::op::max{}, l0, l1);
    p.add_instruction(migraphx::op::max{}, curr_max, l2);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{7, 8, 9};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(min_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l0       = p.add_literal(migraphx::literal{s, {1, 4, 3}});
    auto l1       = p.add_literal(migraphx::literal{s, {2, 8, 6}});
    auto l2       = p.add_literal(migraphx::literal{s, {7, 5, 9}});
    auto curr_min = p.add_instruction(migraphx::op::min{}, l0, l1);
    p.add_instruction(migraphx::op::min{}, curr_min, l2);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 4, 3};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(rnn_forward)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> wf_data{0.4691,
                               0.3185,
                               -0.2227,
                               0.4423,
                               -0.0609,
                               -0.2803,
                               0.1744,
                               0.3146,
                               0.4049,
                               -0.3973,
                               -0.0890,
                               -0.1636};
    std::vector<float> rf_data{-0.0456,
                               0.1061,
                               0.1574,
                               -0.4928,
                               -0.4300,
                               -0.1909,
                               -0.0225,
                               -0.2668,
                               0.1840,
                               -0.4453,
                               -0.4896,
                               0.1302,
                               -0.0929,
                               0.3545,
                               -0.4981,
                               0.0616};
    std::vector<float> biasf_data{
        -0.4938, 0.4355, -0.3186, 0.2094, 0.1037, -0.1071, 0.4504, -0.3990};
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    float clip          = 0.0f;
    {
        std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        auto seq = p.add_literal(migraphx::literal{in_shape, input});

        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});

        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        auto w = p.add_literal(migraphx::literal{w_shape, wf_data});
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        auto r = p.add_literal(migraphx::literal{r_shape, rf_data});
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        auto bias = p.add_literal(migraphx::literal{b_shape, biasf_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn::forward,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027,
                                        0.03445704,
                                        0.19167931,
                                        -0.3946827,
                                        -0.30889652,
                                        -0.22276389,
                                        0.44193283,
                                        -0.16477929,
                                        -0.11893477};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    {
        std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        auto seq = p.add_literal(migraphx::literal{in_shape, input});

        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});

        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        auto w = p.add_literal(migraphx::literal{w_shape, wf_data});
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        auto r = p.add_literal(migraphx::literal{r_shape, rf_data});
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        auto bias = p.add_literal(migraphx::literal{b_shape, biasf_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn::forward, clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);

        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477};
        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }
}

TEST_CASE(rnn_reverse)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> wr_data{-0.0296,
                               -0.1341,
                               0.1761,
                               -0.2325,
                               -0.0717,
                               0.1852,
                               0.2720,
                               0.1471,
                               -0.1097,
                               0.3363,
                               -0.0587,
                               -0.2302};
    std::vector<float> rr_data{0.2528,
                               -0.2333,
                               0.3973,
                               0.1593,
                               -0.0388,
                               0.1702,
                               0.3829,
                               -0.0712,
                               -0.1668,
                               0.3074,
                               -0.2854,
                               0.4049,
                               -0.3737,
                               -0.1051,
                               0.4482,
                               -0.2841};
    std::vector<float> biasr_data{-0.3188, 0.1341, -0.4446, 0.1389, 0.3117, 0.3664, 0.2352, 0.2552};
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    float clip          = 0.0f;
    {
        std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        auto seq = p.add_literal(migraphx::literal{in_shape, input});

        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});

        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        auto w = p.add_literal(migraphx::literal{w_shape, wr_data});
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        auto r = p.add_literal(migraphx::literal{r_shape, rr_data});
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        auto bias = p.add_literal(migraphx::literal{b_shape, biasr_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn::reverse,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.29385301,
                                        0.16796815,
                                        0.51075965,
                                        0.40258689,
                                        -0.13818839,
                                        0.44124447,
                                        0.14365635,
                                        0.14803654,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    {
        std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        auto seq = p.add_literal(migraphx::literal{in_shape, input});

        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});

        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        auto w = p.add_literal(migraphx::literal{w_shape, wr_data});
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        auto r = p.add_literal(migraphx::literal{r_shape, rr_data});
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        auto bias = p.add_literal(migraphx::literal{b_shape, biasr_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn::reverse, clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);

        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{-0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};
        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }
}

TEST_CASE(rnn_bidirectional)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    std::vector<float> wf_data{0.4691,
                               0.3185,
                               -0.2227,
                               0.4423,
                               -0.0609,
                               -0.2803,
                               0.1744,
                               0.3146,
                               0.4049,
                               -0.3973,
                               -0.0890,
                               -0.1636};
    std::vector<float> wr_data{-0.0296,
                               -0.1341,
                               0.1761,
                               -0.2325,
                               -0.0717,
                               0.1852,
                               0.2720,
                               0.1471,
                               -0.1097,
                               0.3363,
                               -0.0587,
                               -0.2302};
    std::vector<float> rf_data{-0.0456,
                               0.1061,
                               0.1574,
                               -0.4928,
                               -0.4300,
                               -0.1909,
                               -0.0225,
                               -0.2668,
                               0.1840,
                               -0.4453,
                               -0.4896,
                               0.1302,
                               -0.0929,
                               0.3545,
                               -0.4981,
                               0.0616};
    std::vector<float> rr_data{0.2528,
                               -0.2333,
                               0.3973,
                               0.1593,
                               -0.0388,
                               0.1702,
                               0.3829,
                               -0.0712,
                               -0.1668,
                               0.3074,
                               -0.2854,
                               0.4049,
                               -0.3737,
                               -0.1051,
                               0.4482,
                               -0.2841};
    std::vector<float> biasf_data{
        -0.4938, 0.4355, -0.3186, 0.2094, 0.1037, -0.1071, 0.4504, -0.3990};
    std::vector<float> biasr_data{-0.3188, 0.1341, -0.4446, 0.1389, 0.3117, 0.3664, 0.2352, 0.2552};
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    float clip          = 0.0f;
    {
        std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        auto seq = p.add_literal(migraphx::literal{in_shape, input});

        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});

        auto w_data = wf_data;
        w_data.insert(w_data.end(), wr_data.begin(), wr_data.end());
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        auto w = p.add_literal(migraphx::literal{w_shape, w_data});

        auto r_data = rf_data;
        r_data.insert(r_data.end(), rr_data.begin(), rr_data.end());
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        auto r = p.add_literal(migraphx::literal{r_shape, r_data});

        auto bias_data = biasf_data;
        bias_data.insert(bias_data.end(), biasr_data.begin(), biasr_data.end());
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(
            migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn::bidirectional, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.37780784,  0.61055139,  0.55168478,  -0.5888475, -0.37144644, 0.31708236,
            0.13104209,  -0.18736027, -0.29385301, 0.16796815, 0.51075965,  0.40258689,
            -0.13818839, 0.44124447,  0.14365635,  0.14803654, 0.03445704,  0.19167931,
            -0.3946827,  -0.30889652, -0.22276389, 0.44193283, -0.16477929, -0.11893477,
            -0.0070999,  0.46251031,  -0.20639211, 0.37488942, -0.0070999,  0.46251031,
            -0.20639211, 0.37488942};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    {
        std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        auto seq = p.add_literal(migraphx::literal{in_shape, input});

        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});

        auto w_data = wf_data;
        w_data.insert(w_data.end(), wr_data.begin(), wr_data.end());
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        auto w = p.add_literal(migraphx::literal{w_shape, w_data});

        auto r_data = rf_data;
        r_data.insert(r_data.end(), rr_data.begin(), rr_data.end());
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        auto r = p.add_literal(migraphx::literal{r_shape, r_data});

        auto bias_data = biasf_data;
        bias_data.insert(bias_data.end(), biasr_data.begin(), biasr_data.end());
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::rnn{
                hidden_size, {migraphx::op::tanh{}}, migraphx::op::rnn::bidirectional, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);

        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477,
                                                 -0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};

        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }
    {
        std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        auto seq = p.add_literal(migraphx::literal{in_shape, input});

        auto w_data = wf_data;
        w_data.insert(w_data.end(), wr_data.begin(), wr_data.end());
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        auto w = p.add_literal(migraphx::literal{w_shape, w_data});

        auto r_data = rf_data;
        r_data.insert(r_data.end(), rr_data.begin(), rr_data.end());
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        auto r = p.add_literal(migraphx::literal{r_shape, r_data});

        auto bias_data = biasf_data;
        bias_data.insert(bias_data.end(), biasr_data.begin(), biasr_data.end());
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hidden_size,
                                                {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias);

        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477,
                                                 -0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};

        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }
}

TEST_CASE(gru_forward)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;
    // concatenation of hidden states for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::gru::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.27298412, 0.42363745,  -0.09368783, 0.4823072,   -0.02183238, -0.6873896,
            0.16144305,  0.31932795,  0.6104771,   0.79759157,  -0.31791314, 0.5249062,
            0.08800987,  0.46404213,  -0.11872687, -0.26210734, 0.34448293,  -0.0176422,
            0.48523626,  0.60002893,  -0.3969709,  0.43360898,  0.35775262,  0.23280787,
            -0.52179873, -0.21944991, 0.4535257,   -0.13735442, 0.51757574,  0.50380427};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::gru::forward,
                                                clip,
                                                1},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::gru::forward,
                                                clip,
                                                0},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.53291196,
                                        0.50160867,
                                        0.39010462,
                                        0.39292926,
                                        -0.5960838,
                                        -0.38451535,
                                        0.454239,
                                        -0.10620412,
                                        0.6014447,
                                        0.43445644};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 3 args
    {
        migraphx::program p;
        auto seq = p.add_literal(migraphx::literal{in_shape, input});
        auto w   = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r   = p.add_literal(migraphx::literal{r_shape, r_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::gru::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.114674, -0.129581,  -0.218156,  -0.140788,  -0.114242,
                                        -0.346569, 0.321367,   -0.0838253, 0.102097,   0.00232137,
                                        -0.149055, 0.0590743,  -0.0533094, -0.0446122, -0.112588,
                                        0.0153261, 0.168883,   -0.326836,  0.0843562,  0.160872,
                                        -0.232523, 0.00214573, 0.231693,   -0.160475,  -0.518952,
                                        0.0467166, 0.12327,    -0.374162,  0.137778,   0.251976};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 4 args (bias is used)
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::gru::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.273619, 0.0931375, -0.104717,  0.0203752, -0.0797887,
                                        -0.493948, 0.472118,  -0.0336318, 0.332706,  0.0182268,
                                        -0.341684, 0.38063,   0.0589275,  0.2644,    -0.115737,
                                        -0.152324, 0.442277,  -0.201626,  0.408909,  0.12905,
                                        -0.416866, 0.377186,  0.32922,    0.162214,  -0.519973,
                                        -0.140072, 0.465076,  -0.229563,  0.500164,  0.195166};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 4 args (ih is used)
    {
        migraphx::program p;
        auto seq = p.add_literal(migraphx::literal{in_shape, input});
        auto w   = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r   = p.add_literal(migraphx::literal{r_shape, r_data});
        auto ih  = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto und = p.add_instruction(migraphx::op::undefined{});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::gru::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          und,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.0801064, 0.27025,   -0.20704,   0.333579,   -0.0452438,
                                        -0.56265,   0.061061,  0.262172,   0.405193,   0.775226,
                                        -0.100683,  0.258729,  -0.0187297, 0.215815,   -0.108936,
                                        -0.0941018, 0.129665,  -0.159421,  0.190636,   0.597412,
                                        -0.197,     0.0885705, 0.269396,   -0.0414511, -0.515137,
                                        -0.03075,   0.158326,  -0.296488,  0.177983,   0.519498};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // no activation function specified, so default is used.
    {
        migraphx::program p;
        auto seq       = p.add_literal(migraphx::literal{in_shape, input});
        auto w         = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r         = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = p.add_instruction(migraphx::op::undefined{});
        auto ih        = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = p.add_instruction(
            migraphx::op::gru{hidden_size, {}, migraphx::op::gru::forward, clip, 1},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 1 activation function (sigmoid) specified
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(
            migraphx::op::gru{
                hidden_size, {migraphx::op::sigmoid{}}, migraphx::op::gru::forward, clip, 1},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.26905832, 0.5669211,  0.20464146, 0.67195725, 0.24752215,
                                        0.11411376, 0.12353572, 0.4245067,  0.73908687, 0.8644615,
                                        0.34754312, 0.61424744, 0.36769435, 0.6499579,  0.3168031,
                                        0.3296533,  0.3055136,  0.42514813, 0.6851256,  0.7967266,
                                        0.35652235, 0.6033026,  0.52634895, 0.5815402,  0.3001663,
                                        0.39814138, 0.4354002,  0.4310627,  0.6708563,  0.7509278};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 1 activation function (tanh) specified
    {
        migraphx::program p;
        auto seq       = p.add_literal(migraphx::literal{in_shape, input});
        auto w         = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r         = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = p.add_instruction(migraphx::op::undefined{});
        auto ih        = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = p.add_instruction(
            migraphx::op::gru{
                hidden_size, {migraphx::op::tanh{}}, migraphx::op::gru::forward, clip, 1},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.49333298,
                                        -0.06104589,
                                        0.5629142,
                                        -0.97955984,
                                        -0.9314696,
                                        -0.03033514,
                                        0.5280315,
                                        -0.27354342,
                                        0.65615714,
                                        0.53612584};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // seq length of 1
    {
        migraphx::program p;
        seq_len = 1;
        migraphx::shape in_shape_one{migraphx::shape::float_type,
                                     {seq_len, batch_size, input_size}};
        std::vector<float> input_one{-0.8432, -0.9887, 1.3041, -2.6430, -0.3306, -0.8504};
        auto seq  = p.add_literal(migraphx::literal{in_shape_one, input_one});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::gru::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.27298412,
                                        0.42363745,
                                        -0.09368783,
                                        0.4823072,
                                        -0.02183238,
                                        -0.6873896,
                                        0.16144305,
                                        0.31932795,
                                        0.6104771,
                                        0.79759157};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_reverse)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;

    // concatenation of hidden states for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::gru::reverse,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.263403, 0.317655,  -0.00634162, 0.200443, -0.349125,
                                        -0.600874, 0.542386,  -0.0856531,  0.55703,  0.54711,
                                        -0.276245, 0.521348,  0.302874,    0.394353, -0.334369,
                                        -0.187861, 0.213553,  -0.0708377,  0.545435, 0.654301,
                                        -0.329512, 0.476095,  0.284044,    0.392077, -0.369226,
                                        -0.3275,   -0.027301, 0.143774,    0.655686, 0.782831};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::gru::reverse,
                                                clip,
                                                1},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.263403,
                                        0.317655,
                                        -0.00634162,
                                        0.200443,
                                        -0.349125,
                                        -0.600874,
                                        0.542386,
                                        -0.0856531,
                                        0.55703,
                                        0.54711};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::gru::reverse,
                                                clip,
                                                0},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.388654,
                                        0.384975,
                                        0.0179455,
                                        0.350101,
                                        -0.456872,
                                        -0.690085,
                                        0.534512,
                                        -0.0558191,
                                        0.646604,
                                        0.463943};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // no activation function specified, so default is used.
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size, {}, migraphx::op::gru::reverse, clip, 1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.263403, 0.317655,  -0.00634162, 0.200443, -0.349125,
                                        -0.600874, 0.542386,  -0.0856531,  0.55703,  0.54711,
                                        -0.276245, 0.521348,  0.302874,    0.394353, -0.334369,
                                        -0.187861, 0.213553,  -0.0708377,  0.545435, 0.654301,
                                        -0.329512, 0.476095,  0.284044,    0.392077, -0.369226,
                                        -0.3275,   -0.027301, 0.143774,    0.655686, 0.782831};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // seq length of 1
    {
        migraphx::program p;
        seq_len = 1;
        migraphx::shape in_shape_one{migraphx::shape::float_type,
                                     {seq_len, batch_size, input_size}};
        std::vector<float> input_one{-0.8432, -0.9887, 1.3041, -2.6430, -0.3306, -0.8504};
        auto seq  = p.add_literal(migraphx::literal{in_shape_one, input_one});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::gru::reverse,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.272984,
                                        0.423637,
                                        -0.0936878,
                                        0.482307,
                                        -0.0218324,
                                        -0.68739,
                                        0.161443,
                                        0.319328,
                                        0.610477,
                                        0.797592};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_bidirectional)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3809,  0.4283,  0.2294,  -0.1018, -0.1226, -0.0037, 0.2449,  -0.2712, -0.1418,
        0.1363,  -0.3453, -0.0693, -0.2281, 0.2699,  -0.2024, -0.3085, -0.3338, 0.4109,
        0.2605,  -0.1019, -0.2813, 0.3323,  -0.1590, 0.0788,  -0.3535, 0.0397,  0.2732,
        0.2906,  0.0519,  0.3617,  -0.2664, 0.1441,  0.0464,  -0.1057, 0.2204,  -0.3294,
        0.3670,  0.1411,  0.3852,  0.3572,  0.3918,  0.0483,  -0.3906, -0.2841, -0.2778,

        -0.4272, 0.2335,  -0.1811, -0.3885, -0.1279, 0.1000,  0.0206,  -0.3284, -0.0353,
        0.1197,  0.1190,  0.3862,  0.0965,  -0.0492, 0.2657,  -0.1430, 0.0597,  0.1408,
        -0.0315, 0.1248,  0.0751,  0.3838,  0.3020,  0.0515,  0.2375,  -0.4255, 0.1714,
        -0.0432, 0.3447,  -0.2441, -0.3989, -0.3428, -0.4204, -0.4080, -0.2683, -0.0996,
        -0.1685, -0.0532, -0.1258, 0.1663,  -0.3526, -0.3915, -0.1721, 0.1292,  -0.2279};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        -0.2683, 0.0699,  -0.4021, -0.1379, 0.0042,  -0.2447, 0.4006,  0.0270,  -0.0446, 0.1063,
        0.1381,  0.1310,  -0.3596, 0.3869,  0.3929,  0.2750,  0.0890,  0.3069,  -0.1691, -0.2194,
        -0.1066, 0.3187,  -0.4369, -0.0603, -0.0834, -0.1182, -0.2047, 0.3253,  -0.2931, 0.2082,
        0.0424,  0.1111,  -0.2773, -0.0279, -0.0869, 0.1413,  -0.4227, -0.3672, 0.4137,  0.0609,
        0.4223,  -0.4032, 0.2945,  0.3600,  0.3345,  -0.3880, -0.0192, -0.0090, -0.2648, 0.4339,
        -0.0155, 0.4437,  -0.1766, 0.1957,  0.2475,  0.3773,  -0.2710, 0.3289,  -0.2077, -0.2534,
        -0.0832, -0.1632, 0.0728,  0.2520,  0.4153,  0.1659,  -0.4342, 0.0541,  0.1812,  -0.2305,
        0.4440,  0.0946,  0.0410,  -0.4381, -0.3161, 0.3906,  -0.3958, -0.4238, 0.1975,  0.3440,
        0.1437,  -0.0568, 0.1492,  -0.4248, -0.3304, 0.2786,  -0.1328, -0.3740, -0.3566, 0.3074,
        0.0924,  0.2684,  -0.1527, 0.1826,  0.2424,  0.2002,  0.3479,  -0.1089, 0.3472,  -0.3677,
        -0.4231, -0.0798, -0.3709, 0.3924,  0.2774,  -0.3690, -0.0233, 0.2845,  0.1969,  0.1618,
        -0.3742, -0.3619, 0.2925,  -0.1838, -0.1495, -0.3747, 0.0341,  -0.4243, -0.0732, -0.3997,
        0.2139,  0.2425,  0.4171,  -0.3358, 0.3534,  0.0938,  -0.0582, -0.2681, -0.4293, 0.1027,
        0.4101,  0.2641,  -0.4110, -0.1681, 0.3582,  -0.2089, 0.0852,  0.0963,  0.3866,  0.1955,
        -0.2174, 0.1996,  -0.2252, 0.1748,  0.1833,  -0.3155, 0.2567,  -0.4387, 0.3402,  0.0599};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        -0.1582, -0.0826, 0.4008,  0.0118,  0.2511,  0.1900,  -0.2838, 0.2549,  -0.2484, 0.2363,
        -0.4083, -0.0295, -0.1161, 0.1211,  0.2509,  -0.1414, -0.2628, -0.2992, 0.1517,  0.1817,
        -0.2783, 0.3183,  -0.1629, -0.3108, -0.3418, 0.0411,  0.2203,  0.2187,  -0.2990, -0.0416,
        0.0209,  -0.1024, 0.4443,  -0.4420, -0.0330, -0.3591, -0.2990, 0.2167,  0.1395,  0.2317,
        0.1318,  0.1909,  -0.3615, 0.1953,  -0.2582, -0.2217, 0.3723,  0.1458,  0.2630,  -0.0377,
        0.1754,  0.0800,  -0.3964, -0.3247, 0.4219,  -0.0900, 0.3553,  0.2614,  -0.1298, -0.1124};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{-0.0468, 0.5691,  -0.0882, 0.8340,  0.1483, -0.3902, -0.5348,
                               0.4178,  1.0175,  0.9212,  -0.0468, 0.5691, -0.0882, 0.8340,
                               0.1483,  -0.3902, -0.5348, 0.4178,  1.0175, 0.9212};

    float clip = 0.0f;

    // concatenation of hidden states for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::gru::bidirectional,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352243, 0.0146756,  0.00570925, 0.152446,  0.208683,   0.214342,   -0.0454273,
            -0.135177, -0.0800739, 0.903659,   0.0248217, 0.435231,   -0.144448,  0.101531,
            -0.111305, 0.381317,   0.468983,   0.230557,  0.348021,   0.180229,   -0.0930435,
            0.174108,  -0.063834,  0.0909285,  0.22759,   -0.221983,  -0.139656,  -0.0938906,
            -0.247681, 0.69647,    -0.159396,  0.299061,  -0.116652,  0.238649,   0.109945,
            0.192866,  0.307073,   0.191113,   0.658287,  -0.0340374, -0.0959787, 0.0794681,
            0.241526,  0.321104,   0.00693533, -0.311839, -0.12802,   -0.16643,   -0.393849,
            0.648851,  -0.395918,  0.231694,   -0.160503, 0.383289,   0.0879262,  -0.0254665,
            0.079043,  0.322652,   0.752701,   0.243775};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::gru::bidirectional,
                                                clip,
                                                1},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.0959787, 0.0794681, 0.241526,  0.321104,  0.00693533,
                                        -0.311839,  -0.12802,  -0.16643,  -0.393849, 0.648851,
                                        0.0248217,  0.435231,  -0.144448, 0.101531,  -0.111305,
                                        0.381317,   0.468983,  0.230557,  0.348021,  0.180229};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::gru::bidirectional,
                                                clip,
                                                0},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.09280921, 0.18506107, 0.32247013, 0.17034212, -0.00115255, -0.29865006, -0.04513004,
            -0.10688055, -0.4767866, 0.6317833,  0.00286336, 0.53692746,  -0.00617076, 0.04564289,
            -0.18030001, 0.39584228, 0.53879917, 0.384983,   0.2759448,   0.11611474};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // // 3 args
    // {
    //     migraphx::program p;
    //     auto seq = p.add_literal(migraphx::literal{in_shape, input});
    //     auto w = p.add_literal(migraphx::literal{w_shape, w_data});
    //     auto r = p.add_literal(migraphx::literal{r_shape, r_data});
    //     p.add_instruction(migraphx::op::gru{hidden_size,
    //                                         {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
    //                                         migraphx::op::gru::forward,
    //                                         clip,
    //                                         1},
    //                       seq,
    //                       w,
    //                       r);

    //     p.compile(migraphx::cpu::target{});
    //     auto hs_concat = p.eval({});
    //     std::vector<float> hs_data;
    //     hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    //     std::vector<float> hs_data_gold{
    //         -0.114674, -0.129581, -0.218156, -0.140788, -0.114242,
    //         -0.346569, 0.321367, -0.0838253, 0.102097, 0.00232137,
    //         -0.149055, 0.0590743, -0.0533094, -0.0446122, -0.112588,
    //         0.0153261, 0.168883, -0.326836, 0.0843562, 0.160872,
    //         -0.232523, 0.00214573, 0.231693, -0.160475, -0.518952,
    //         0.0467166, 0.12327, -0.374162, 0.137778, 0.251976};

    //     EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    // }

    // // 4 args (bias is used)
    // {
    //     migraphx::program p;
    //     auto seq = p.add_literal(migraphx::literal{in_shape, input});
    //     auto w = p.add_literal(migraphx::literal{w_shape, w_data});
    //     auto r = p.add_literal(migraphx::literal{r_shape, r_data});
    //     auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
    //     p.add_instruction(migraphx::op::gru{hidden_size,
    //                                         {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
    //                                         migraphx::op::gru::forward,
    //                                         clip,
    //                                         1},
    //                       seq,
    //                       w,
    //                       r,
    //                       bias);

    //     p.compile(migraphx::cpu::target{});
    //     auto hs_concat = p.eval({});
    //     std::vector<float> hs_data;
    //     hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    //     std::vector<float> hs_data_gold{
    //         -0.273619, 0.0931375, -0.104717, 0.0203752, -0.0797887,
    //         -0.493948, 0.472118, -0.0336318, 0.332706, 0.0182268,
    //         -0.341684, 0.38063, 0.0589275, 0.2644, -0.115737,
    //         -0.152324, 0.442277, -0.201626, 0.408909, 0.12905,
    //         -0.416866, 0.377186, 0.32922, 0.162214, -0.519973,
    //         -0.140072, 0.465076, -0.229563, 0.500164, 0.195166};

    //     EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    // }

    // // 4 args (ih is used)
    // {
    //     migraphx::program p;
    //     auto seq = p.add_literal(migraphx::literal{in_shape, input});
    //     auto w = p.add_literal(migraphx::literal{w_shape, w_data});
    //     auto r = p.add_literal(migraphx::literal{r_shape, r_data});
    //     auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});
    //     auto und  = p.add_instruction(migraphx::op::undefined{});
    //     p.add_instruction(migraphx::op::gru{hidden_size,
    //                                         {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
    //                                         migraphx::op::gru::forward,
    //                                         clip,
    //                                         1},
    //                       seq,
    //                       w,
    //                       r,
    //                       und,
    //                       und,
    //                       ih);

    //     p.compile(migraphx::cpu::target{});
    //     auto hs_concat = p.eval({});
    //     std::vector<float> hs_data;
    //     hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    //     std::vector<float> hs_data_gold{
    //         -0.0801064, 0.27025, -0.20704, 0.333579, -0.0452438,
    //         -0.56265, 0.061061, 0.262172, 0.405193, 0.775226,
    //         -0.100683, 0.258729, -0.0187297, 0.215815, -0.108936,
    //         -0.0941018, 0.129665, -0.159421, 0.190636, 0.597412,
    //         -0.197, 0.0885705, 0.269396, -0.0414511, -0.515137,
    //         -0.03075, 0.158326, -0.296488, 0.177983, 0.519498};

    //     EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    // }

    // // no activation function specified, so default is used.
    // {
    //     migraphx::program p;
    //     auto seq = p.add_literal(migraphx::literal{in_shape, input});
    //     auto w = p.add_literal(migraphx::literal{w_shape, w_data});
    //     auto r = p.add_literal(migraphx::literal{r_shape, r_data});
    //     auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
    //     auto und  = p.add_instruction(migraphx::op::undefined{});
    //     auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});
    //     auto concat_hs = p.add_instruction(migraphx::op::gru{hidden_size,
    //                                         {},
    //                                         migraphx::op::gru::forward,
    //                                         clip,
    //                                         1},
    //                       seq,
    //                       w,
    //                       r,
    //                       bias,
    //                       und,
    //                       ih);
    //     p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
    //     p.compile(migraphx::cpu::target{});
    //     auto hs_concat = p.eval({});
    //     std::vector<float> hs_data;
    //     hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    //     std::vector<float> hs_data_gold{
    //         -0.3969709 ,  0.43360898,  0.35775262,  0.23280787, -0.52179873,
    //         -0.21944991,  0.4535257 , -0.13735442,  0.51757574,  0.50380427};

    //     EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    // }

    // // 1 activation function (sigmoid) specified
    // {
    //     migraphx::program p;
    //     auto seq = p.add_literal(migraphx::literal{in_shape, input});
    //     auto w = p.add_literal(migraphx::literal{w_shape, w_data});
    //     auto r = p.add_literal(migraphx::literal{r_shape, r_data});
    //     auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
    //     auto und  = p.add_instruction(migraphx::op::undefined{});
    //     auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});
    //     p.add_instruction(migraphx::op::gru{hidden_size,
    //                                         {migraphx::op::sigmoid{}},
    //                                         migraphx::op::gru::forward,
    //                                         clip,
    //                                         1},
    //                       seq,
    //                       w,
    //                       r,
    //                       bias,
    //                       und,
    //                       ih);
    //     p.compile(migraphx::cpu::target{});
    //     auto hs_concat = p.eval({});
    //     std::vector<float> hs_data;
    //     hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    //     std::vector<float> hs_data_gold{
    //         0.26905832, 0.5669211 , 0.20464146, 0.67195725, 0.24752215,
    //         0.11411376, 0.12353572, 0.4245067 , 0.73908687, 0.8644615,
    //         0.34754312, 0.61424744, 0.36769435, 0.6499579 , 0.3168031,
    //         0.3296533 , 0.3055136 , 0.42514813, 0.6851256 , 0.7967266,
    //         0.35652235, 0.6033026 , 0.52634895, 0.5815402 , 0.3001663,
    //         0.39814138, 0.4354002 , 0.4310627 , 0.6708563 , 0.7509278};

    //     EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    // }

    // // 1 activation function (tanh) specified
    // {
    //     migraphx::program p;
    //     auto seq = p.add_literal(migraphx::literal{in_shape, input});
    //     auto w = p.add_literal(migraphx::literal{w_shape, w_data});
    //     auto r = p.add_literal(migraphx::literal{r_shape, r_data});
    //     auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
    //     auto und  = p.add_instruction(migraphx::op::undefined{});
    //     auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});
    //     auto concat_hs = p.add_instruction(migraphx::op::gru{hidden_size,
    //                                         {migraphx::op::tanh{}},
    //                                         migraphx::op::gru::forward,
    //                                         clip,
    //                                         1},
    //                       seq,
    //                       w,
    //                       r,
    //                       bias,
    //                       und,
    //                       ih);
    //     p.add_instruction(migraphx::op::gru_last_output{}, concat_hs);
    //     p.compile(migraphx::cpu::target{});
    //     auto hs_concat = p.eval({});
    //     std::vector<float> hs_data;
    //     hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    //     std::vector<float> hs_data_gold{
    //         -0.49333298, -0.06104589,  0.5629142,  -0.97955984, -0.9314696,
    //         -0.03033514,  0.5280315,  -0.27354342,  0.65615714,  0.53612584};

    //     EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    // }

    // // seq length of 1
    // {
    //     migraphx::program p;
    //     seq_len = 1;
    //     migraphx::shape in_shape_one{migraphx::shape::float_type, {seq_len, batch_size,
    //     input_size}}; std::vector<float> input_one{-0.8432, -0.9887,  1.3041, -2.6430, -0.3306,
    //     -0.8504}; auto seq = p.add_literal(migraphx::literal{in_shape_one, input_one}); auto w =
    //     p.add_literal(migraphx::literal{w_shape, w_data}); auto r =
    //     p.add_literal(migraphx::literal{r_shape, r_data}); auto bias =
    //     p.add_literal(migraphx::literal{b_shape, bias_data}); auto und  =
    //     p.add_instruction(migraphx::op::undefined{}); auto ih =
    //     p.add_literal(migraphx::literal{ih_shape, ih_data});
    //     p.add_instruction(migraphx::op::gru{hidden_size,
    //                                         {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
    //                                         migraphx::op::gru::forward,
    //                                         clip,
    //                                         1},
    //                       seq,
    //                       w,
    //                       r,
    //                       bias,
    //                       und,
    //                       ih);

    //     p.compile(migraphx::cpu::target{});
    //     auto hs_concat = p.eval({});
    //     std::vector<float> hs_data;
    //     hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

    //     std::vector<float> hs_data_gold{
    //         -0.27298412,  0.42363745, -0.09368783,  0.4823072 , -0.02183238,
    //         -0.6873896 ,  0.16144305,  0.31932795,  0.6104771 ,  0.79759157};

    //     EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    // }
}

TEST_CASE(pad_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = p.add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    p.add_instruction(migraphx::op::pad{{1, 1, 1, 1}}, l0);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
