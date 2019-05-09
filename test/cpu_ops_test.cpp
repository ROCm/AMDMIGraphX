#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"
#include <migraphx/half.hpp>

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

    {
        migraphx::program p;

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = p.add_literal(migraphx::literal{s, data});
        // scalar index
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{0};
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<float> res_data{};
        std::vector<float> golden = {0.5f, 3.5f, 6.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }

    {
        migraphx::program p;

        std::vector<float> data(3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto a0 = p.add_literal(migraphx::literal{s, data});
        // scalar index
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{0};
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<float> res_data{};
        std::vector<float> golden = {0.5f};
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
    p.add_instruction(migraphx::op::broadcast{axis, l1->get_shape().lens()}, l2);
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
        auto l3 = p.add_instruction(migraphx::op::broadcast{axis, l1->get_shape().lens()}, l2);
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

TEST_CASE(lrn_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {1, 5, 1, 1}};
    auto l = p.add_literal(migraphx::literal{s, {-2.0f, 1.0f, 0.f, 1.0f, 2.0f}});
    p.add_instruction(migraphx::op::lrn{0.0001, 0.75, 1, 5}, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(5);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2 / 1.000075, 1 / 1.00009, 0 / 1.000145, 1 / 1.00009, 2 / 1.000075};
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
    auto scaled_tensor = p.add_instruction(migraphx::op::scalar{s.lens()}, scale_val);
    auto img_scaled    = p.add_instruction(migraphx::op::mul{}, img, scaled_tensor);
    auto bias_vals     = p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0.01, 0.02, 0.03}});
    auto bias_bcast = p.add_instruction(migraphx::op::broadcast{1, s.lens()}, bias_vals);
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

TEST_CASE(logsoftmax_test_axis_0)
{
    migraphx::program p;
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -2.71138556, -5.85030702, -3.74063578, -4.22915517, -6.15821977, -5.96072346, -3.57208097,
        -5.78313166, -5.51435497, -3.67224195, -3.88393048, -2.57061599, -5.54431083, -6.27880025,
        -5.1878749,  -6.1318955,  -5.29178545, -4.22537886, -3.75693516, -7.07047099, -4.45763333,
        -4.66281846, -6.18290503, -4.11886536, -6.17408292, -4.18030052, -4.64570814, -4.64354473,
        -3.06629525, -3.80807681, -4.69162374, -5.53605222, -3.20969275, -4.82645674, -6.63942356,
        -4.73634471, -3.86003866, -5.32738981, -4.22249802, -4.51258693, -2.41455206, -3.48343199,
        -5.86215889, -4.93435935, -4.83713408, -2.97471885, -2.16666459, -3.69133151, -4.71640968,
        -5.64652924, -3.60709827, -5.87967748, -3.8809403,  -4.33917815};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = p.add_literal(migraphx::literal{a_shape, a});
    int axis = 0;
    p.add_instruction(migraphx::op::logsoftmax{axis}, al);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_1)
{
    migraphx::program p;
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -1.77931988, -4.91824134, -2.80857010, -3.29708949, -5.22615409, -5.02865778, -2.64001529,
        -4.85106598, -4.58228929, -2.74017627, -2.95186480, -1.63855031, -4.61224515, -5.34673457,
        -4.25580922, -5.19982982, -4.35971977, -3.29331318, -2.82486948, -6.13840531, -3.52556765,
        -3.73075278, -5.25083935, -3.18679968, -5.24201724, -3.24823484, -3.71364246, -4.14309917,
        -2.56584969, -3.30763125, -4.19117818, -5.03560666, -2.70924719, -4.32601118, -6.13897800,
        -4.23589915, -3.35959310, -4.82694425, -3.72205246, -4.01214137, -1.91410650, -2.98298643,
        -5.36171333, -4.43391379, -4.33668852, -2.47427329, -1.66621903, -3.19088595, -4.21596412,
        -5.14608368, -3.10665271, -5.37923192, -3.38049474, -3.83873259};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = p.add_literal(migraphx::literal{a_shape, a});
    int axis = 1;
    p.add_instruction(migraphx::op::logsoftmax{axis}, al);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_2)
{
    migraphx::program p;
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.79763715, -3.93655861, -1.82688737, -2.31540676, -4.24447136, -4.04697505, -1.65833256,
        -3.86938325, -3.60060656, -1.81223672, -2.02392525, -0.71061076, -3.68430560, -4.41879502,
        -3.32786967, -4.27189027, -3.43178022, -2.36537363, -1.35498658, -4.66852241, -2.05568475,
        -2.26086988, -3.78095645, -1.71691678, -3.77213434, -1.77835194, -2.24375956, -2.74631770,
        -1.16906822, -1.91084978, -2.79439671, -3.63882519, -1.31246572, -2.92922971, -4.74219653,
        -2.83911768, -2.19738500, -3.66473615, -2.55984436, -2.84993327, -0.75189840, -1.82077833,
        -4.19950523, -3.27170569, -3.17448042, -1.65286841, -0.84481415, -2.36948107, -3.39455924,
        -4.32467880, -2.28524783, -4.55782704, -2.55908986, -3.01732771};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = p.add_literal(migraphx::literal{a_shape, a});
    int axis = 2;
    p.add_instruction(migraphx::op::logsoftmax{axis}, al);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_3)
{
    migraphx::program p;
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.33690375, -3.47582521, -1.36615397, -0.27936556, -2.20843016, -2.01093385, -0.22551114,
        -2.43656183, -2.16778514, -1.57241522, -1.78410375, -0.47078926, -1.06745881, -1.80194823,
        -0.71102288, -2.30719726, -1.46708721, -0.40068062, -0.42698261, -3.74051844, -1.12768078,
        -1.07891856, -2.59900513, -0.53496546, -2.56139951, -0.56761711, -1.03302473, -2.09771276,
        -0.52046328, -1.26224484, -1.76322959, -2.60765807, -0.28129860, -0.81424303, -2.62720985,
        -0.72413100, -0.65570381, -2.12305496, -1.01816317, -2.48063402, -0.38259915, -1.45147908,
        -1.84310238, -0.91530284, -0.81807757, -1.31692881, -0.50887455, -2.03354147, -1.48767160,
        -2.41779116, -0.37836019, -2.56853147, -0.56979429, -1.02803214};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = p.add_literal(migraphx::literal{a_shape, a});
    int axis = 3;
    p.add_instruction(migraphx::op::logsoftmax{axis}, al);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_4)
{
    migraphx::program p;
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = p.add_literal(migraphx::literal{a_shape, a});
    int axis = 4;
    p.add_instruction(migraphx::op::logsoftmax{axis}, al);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector;
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

TEST_CASE(quant_conv2d_test)
{
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = p.add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = p.add_literal(migraphx::literal{c_shape, c});

    p.add_instruction(migraphx::op::quant_convolution{}, al, cl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> s = {10197,
                            10548,
                            11601,
                            11952,
                            25506,
                            26586,
                            29826,
                            30906,
                            27045,
                            27396,
                            28449,
                            28800,
                            77346,
                            78426,
                            81666,
                            82746};

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(quant_conv2d_test_default_mode)
{
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = p.add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = p.add_literal(migraphx::literal{c_shape, c});

    p.add_instruction(
        migraphx::op::quant_convolution{{{0, 0}}, {{1, 1}}, {{1, 1}}, migraphx::op::same}, al, cl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> s = {
        10197, 10548, 6939,  3420,  11601, 11952, 7839,  3852,  7383,  7590,  4953,  2421,  3480,
        3570,  2316,  1125,  25506, 26586, 17874, 9009,  29826, 30906, 20718, 10413, 20505, 21198,
        14187, 7119,  10527, 10860, 7257,  3636,  27045, 27396, 17739, 8604,  28449, 28800, 18639,
        9036,  17319, 17526, 11289, 5445,  7800,  7890,  5052,  2421,  77346, 78426, 52002, 25857,
        81666, 82746, 54846, 27261, 53769, 54462, 36075, 17919, 26511, 26844, 17769, 8820};

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(quant_conv2d_test_valid_mode)
{
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = p.add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = p.add_literal(migraphx::literal{c_shape, c});

    p.add_instruction(
        migraphx::op::quant_convolution{{{0, 0}}, {{1, 1}}, {{1, 1}}, migraphx::op::valid}, al, cl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> s = {10197,
                            10548,
                            11601,
                            11952,
                            25506,
                            26586,
                            29826,
                            30906,
                            27045,
                            27396,
                            28449,
                            28800,
                            77346,
                            78426,
                            81666,
                            82746};

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(quant_conv2d_padding_test)
{
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = p.add_literal(migraphx::literal{c_shape, c});
    p.add_instruction(migraphx::op::quant_convolution{{{1, 1}}, {{1, 1}}}, al, cl);
    p.compile(migraphx::cpu::target{});
    auto result          = p.eval({});
    std::vector<float> s = {
        4521,  6753,  7014,  4635,  6858,  10197, 10548, 6939,  7830,  11601, 11952, 7839,  5007,
        7383,  7590,  4953,  10515, 15987, 16734, 11277, 16821, 25506, 26586, 17874, 19737, 29826,
        30906, 20718, 13593, 20505, 21198, 14187, 13161, 19281, 19542, 12699, 18522, 27045, 27396,
        17739, 19494, 28449, 28800, 18639, 11919, 17319, 17526, 11289, 34707, 51843, 52590, 34893,
        51813, 77346, 78426, 52002, 54729, 81666, 82746, 54846, 36057, 53769, 54462, 36075};

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(quant_conv2d_padding_stride_test)
{
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = p.add_literal(migraphx::literal{c_shape, c});
    p.add_instruction(migraphx::op::quant_convolution{{{1, 1}}, {{2, 2}}}, al, cl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});

    std::vector<float> s = {4521,
                            7014,
                            7830,
                            11952,
                            10515,
                            16734,
                            19737,
                            30906,
                            13161,
                            19542,
                            19494,
                            28800,
                            34707,
                            52590,
                            54729,
                            82746};
    std::vector<float> results_vector;
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

TEST_CASE(fp16_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::half_type, {1}};
    migraphx::half a{1.5};
    migraphx::half b{2.5};
    migraphx::half c{4.0};
    auto l0 = p.add_literal(migraphx::literal{s, {a}});
    auto l1 = p.add_literal(migraphx::literal{s, {b}});
    p.add_instruction(migraphx::op::add{}, l0, l1);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<migraphx::half> results_vector(1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<migraphx::half> gold{c};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(clip_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {-1.0, 0.0, 10.0}});
    migraphx::op::clip op;
    op.max_val = 6.0;
    op.min_val = 0.0;
    p.add_instruction(op, l);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.0, 0.0, 6.0};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
