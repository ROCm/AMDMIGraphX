#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

TEST_CASE(pytorch_conv_bias_test)
{
    migraphx::program p;
    auto l0       = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape().lens()}, l2);
    p.add_instruction(migraphx::op::add{}, l3, l4);

    auto prog = migraphx::parse_onnx("conv.onnx");
    EXPECT(p == prog);
}

TEST_CASE(pytorch_conv_relu_maxpool)
{
    migraphx::program p;
    auto l0       = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape().lens()}, l2);
    auto l5       = p.add_instruction(migraphx::op::add{}, l3, l4);
    auto l6       = p.add_instruction(migraphx::op::relu{}, l5);
    p.add_instruction(migraphx::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l6);

    auto prog = migraphx::parse_onnx("conv_relu_maxpool.onnx");
    EXPECT(p == prog);
}

TEST_CASE(pytorch_conv_bn_relu_maxpool)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1 = p.add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2 = p.add_parameter("2", {migraphx::shape::float_type, {1}});

    auto p3       = p.add_parameter("3", {migraphx::shape::float_type, {1}});
    auto p4       = p.add_parameter("4", {migraphx::shape::float_type, {1}});
    auto p5       = p.add_parameter("5", {migraphx::shape::float_type, {1}});
    auto p6       = p.add_parameter("6", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape().lens()}, l2);
    auto l5       = p.add_instruction(migraphx::op::add{}, l3, l4);
    auto l6 = p.add_instruction(migraphx::op::batch_norm_inference{1.0e-5f}, l5, p3, p4, p5, p6);
    auto l7 = p.add_instruction(migraphx::op::relu{}, l6);
    p.add_instruction(migraphx::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l7);

    auto prog = migraphx::parse_onnx("conv_bn_relu_maxpool.onnx");
    EXPECT(p == prog);
}

TEST_CASE(pytorch_conv_relu_maxpool_x2)
{
    migraphx::program p;
    auto l0       = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraphx::shape::float_type, {5, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraphx::shape::float_type, {5}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape().lens()}, l2);
    auto l5       = p.add_instruction(migraphx::op::add{}, l3, l4);
    auto l6       = p.add_instruction(migraphx::op::relu{}, l5);
    auto l7 = p.add_instruction(migraphx::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l6);

    auto l8  = p.add_parameter("3", {migraphx::shape::float_type, {1, 5, 5, 5}});
    auto l9  = p.add_parameter("4", {migraphx::shape::float_type, {1}});
    auto l10 = p.add_instruction(migraphx::op::convolution{}, l7, l8);
    auto l11 = p.add_instruction(migraphx::op::broadcast{axis, l10->get_shape().lens()}, l9);
    auto l12 = p.add_instruction(migraphx::op::add{}, l10, l11);
    auto l13 = p.add_instruction(migraphx::op::relu{}, l12);
    p.add_instruction(migraphx::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l13);

    auto prog = migraphx::parse_onnx("conv_relu_maxpoolX2.onnx");

    EXPECT(p == prog);
}

TEST_CASE(leaky_relu_test)
{
    migraphx::program p;
    float alpha = 0.01f;
    auto l0     = p.add_parameter("0", {migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::leaky_relu{alpha}, l0);

    auto prog = migraphx::parse_onnx("leaky_relu.onnx");

    EXPECT(p == prog);
}

TEST_CASE(imagescaler_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 16, 16}};
    auto l0        = p.add_parameter("0", s);
    auto scale_val = p.add_literal(0.5f);
    auto bias_vals = p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0.01, 0.02, 0.03}});
    auto scaled_tensor = p.add_instruction(migraphx::op::scalar{s.lens()}, scale_val);
    auto img_scaled    = p.add_instruction(migraphx::op::mul{}, l0, scaled_tensor);
    auto bias_bcast    = p.add_instruction(migraphx::op::broadcast{1, s.lens()}, bias_vals);
    p.add_instruction(migraphx::op::add{}, img_scaled, bias_bcast);

    auto prog = migraphx::parse_onnx("imagescaler_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(globalavgpool_test)
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto op    = migraphx::op::pooling{"average"};
    auto lens  = input->get_shape().lens();
    op.lengths = {lens[2], lens[3]};
    p.add_instruction(op, input);

    auto prog = migraphx::parse_onnx("globalavgpool_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(globalmaxpool_test)
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto op    = migraphx::op::pooling{"max"};
    auto lens  = input->get_shape().lens();
    op.lengths = {lens[2], lens[3]};
    p.add_instruction(op, input);

    auto prog = migraphx::parse_onnx("globalmaxpool_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(transpose_test)
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    std::vector<int64_t> perm{0, 3, 1, 2};
    p.add_instruction(migraphx::op::transpose{perm}, input);

    auto prog = migraphx::parse_onnx("transpose_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(dropout_test)
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}});
    p.add_instruction(migraphx::op::identity{}, input);

    auto prog = migraphx::parse_onnx("dropout_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sum_test)
{
    migraphx::program p;
    auto input0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = p.add_instruction(migraphx::op::add{}, input0, input1);
    p.add_instruction(migraphx::op::add{}, l0, input2);

    auto prog = migraphx::parse_onnx("sum_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(exp_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::exp{}, input);

    auto prog = migraphx::parse_onnx("exp_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(log_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::log{}, input);

    auto prog = migraphx::parse_onnx("log_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sin_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::sin{}, input);

    auto prog = migraphx::parse_onnx("sin_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(cos_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::cos{}, input);

    auto prog = migraphx::parse_onnx("cos_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(tan_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::tan{}, input);

    auto prog = migraphx::parse_onnx("tan_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sinh_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::sinh{}, input);

    auto prog = migraphx::parse_onnx("sinh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(cosh_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    p.add_instruction(migraphx::op::cosh{}, input);

    auto prog = migraphx::parse_onnx("cosh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(tanh_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    p.add_instruction(migraphx::op::tanh{}, input);

    auto prog = migraphx::parse_onnx("tanh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(elu_test)
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::elu{0.01}, input);

    auto prog = migraphx::parse_onnx("elu_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(asin_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::asin{}, input);

    auto prog = migraphx::parse_onnx("asin_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(max_test)
{
    migraphx::program p;
    auto input0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = p.add_instruction(migraphx::op::max{}, input0, input1);
    p.add_instruction(migraphx::op::max{}, l0, input2);

    migraphx::parse_onnx("max_test.onnx");
}

TEST_CASE(acos_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::acos{}, input);

    auto prog = migraphx::parse_onnx("acos_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(min_test)
{
    migraphx::program p;
    auto input0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = p.add_instruction(migraphx::op::min{}, input0, input1);
    p.add_instruction(migraphx::op::min{}, l0, input2);

    migraphx::parse_onnx("min_test.onnx");
}

TEST_CASE(atan_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::atan{}, input);

    auto prog = migraphx::parse_onnx("atan_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(add_bcast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2 = p.add_instruction(migraphx::op::broadcast{1, l0->get_shape().lens()}, l1);
    p.add_instruction(migraphx::op::add{}, l0, l2);

    auto prog = migraphx::parse_onnx("add_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(implicit_add_bcast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4, 1}});
    auto l2 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l0);
    auto l3 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l1);
    p.add_instruction(migraphx::op::add{}, l2, l3);

    auto prog = migraphx::parse_onnx("implicit_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sub_bcast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2 = p.add_instruction(migraphx::op::broadcast{1, l0->get_shape().lens()}, l1);
    p.add_instruction(migraphx::op::sub{}, l0, l2);

    auto prog = migraphx::parse_onnx("sub_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(implicit_sub_bcast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 5}});
    auto l2 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l0);
    auto l3 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l1);
    p.add_instruction(migraphx::op::sub{}, l2, l3);

    auto prog = migraphx::parse_onnx("implicit_sub_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(unknown_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2 = p.add_instruction(migraphx::op::unknown{"Unknown"}, l0, l1);
    p.add_instruction(migraphx::op::unknown{"Unknown"}, l2);
    auto prog = migraphx::parse_onnx("unknown_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(softmax_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    auto r  = p.add_instruction(migraphx::op::reshape{{1, 3, 1, 1}}, l0);
    auto s  = p.add_instruction(migraphx::op::softmax{}, r);
    p.add_instruction(migraphx::op::reshape{{1, 3}}, s);
    auto prog = migraphx::parse_onnx("softmax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reshape_test)
{
    migraphx::program p;
    migraphx::op::reshape op;
    std::vector<int64_t> reshape_dims{3, 8};
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});
    p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, reshape_dims});
    op.dims = reshape_dims;
    p.add_instruction(op, l0);
    p.add_instruction(op, l0);
    auto prog = migraphx::parse_onnx("reshape_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(shape_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3, 4, 5, 6}};
    auto l0 = p.add_parameter("x", s);
    migraphx::shape s_shape{migraphx::shape::int64_type, {4}};
    p.add_literal(s_shape, l0->get_shape().lens());
    auto prog = migraphx::parse_onnx("shape_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gather_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = p.add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, {2, 3}});
    int axis = 1;
    p.add_instruction(migraphx::op::gather{axis}, l0, l1);
    auto prog = migraphx::parse_onnx("gather_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(shape_gather_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {7, 3, 10}});
    auto l1 =
        p.add_literal(migraphx::shape{migraphx::shape::int64_type, {3}}, l0->get_shape().lens());
    migraphx::shape const_shape{migraphx::shape::int32_type, {1}};
    auto l2  = p.add_literal(migraphx::literal{const_shape, {1}});
    int axis = 0;
    p.add_instruction(migraphx::op::gather{axis}, l1, l2);
    auto prog = migraphx::parse_onnx("shape_gather.onnx");

    EXPECT(p == prog);
}

TEST_CASE(flatten_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    p.add_instruction(migraphx::op::flatten{2}, l0);
    p.add_instruction(migraphx::op::flatten{1}, l0);
    auto prog = migraphx::parse_onnx("flatten_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(squeeze_unsqueeze_test)
{
    migraphx::program p;
    std::vector<int64_t> squeeze_axes{0, 2, 3, 5};
    std::vector<int64_t> unsqueeze_axes{0, 1, 3, 5};
    auto l0 =
        p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 1, 1, 2, 1}});
    auto l1 = p.add_instruction(migraphx::op::squeeze{squeeze_axes}, l0);
    p.add_instruction(migraphx::op::unsqueeze{unsqueeze_axes}, l1);
    auto prog = migraphx::parse_onnx("squeeze_unsqueeze_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(concat_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7, 4, 3}});
    p.add_instruction(migraphx::op::concat{0}, l0, l1);
    auto prog = migraphx::parse_onnx("concat_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(slice_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 2}});
    p.add_instruction(migraphx::op::slice{{0, 1}, {1, 0}, {2, 2}}, l0);
    auto prog = migraphx::parse_onnx("slice_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_test)
{
    migraphx::program p;
    p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0, 1, 2}});
    auto prog = migraphx::parse_onnx("constant_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_test_scalar)
{
    migraphx::program p;
    p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1}}, {1}});
    auto prog = migraphx::parse_onnx("constant_scalar.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_fill_test)
{
    {
        migraphx::program p;
        auto l0 = p.add_literal(migraphx::literal{{migraphx::shape::int32_type, {2}}, {2, 3}});
        std::vector<std::size_t> dims(l0->get_shape().elements());
        migraphx::literal ls = l0->get_literal();
        ls.visit([&](auto s) { dims.assign(s.begin(), s.end()); });
        migraphx::shape s{migraphx::shape::float_type, dims};
        std::vector<float> value(s.elements(), 1.0);
        p.add_literal(migraphx::literal{s, value});
        auto prog = migraphx::parse_onnx("const_fill1.onnx");

        EXPECT(p == prog);
    }

    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> value(s.elements(), 1.0);
        p.add_literal(migraphx::literal{s, value});
        auto prog = migraphx::parse_onnx("const_fill2.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(gemm_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 7}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {11, 5}});
    p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {}});
    auto t0    = p.add_instruction(migraphx::op::transpose{{1, 0}}, l0);
    auto t1    = p.add_instruction(migraphx::op::transpose{{1, 0}}, l1);
    auto alpha = 2.f;
    auto beta  = 2.0f;
    p.add_instruction(migraphx::op::dot{alpha, beta}, t0, t1);
    auto prog = migraphx::parse_onnx("gemm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gemm_ex)
{
    migraphx::program p;
    auto l0    = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 6}});
    auto l1    = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 7}});
    auto l2    = p.add_parameter("3", migraphx::shape{migraphx::shape::float_type, {1, 1, 6, 7}});
    auto t0    = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, l0);
    auto alpha = 0.5f;
    auto beta  = 0.8f;
    p.add_instruction(migraphx::op::dot{alpha, beta}, t0, l1, l2);
    auto prog = migraphx::parse_onnx("gemm_test_ex.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gemm_ex_brcst)
{
    migraphx::program p;
    auto l0 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 6}});
    auto l1 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 7}});
    auto l2 = p.add_parameter("3", migraphx::shape{migraphx::shape::float_type, {1, 1, 6, 1}});
    auto t0 = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, l0);
    std::vector<std::size_t> out_lens{1, 1, 6, 7};
    auto t2    = p.add_instruction(migraphx::op::multibroadcast{out_lens}, l2);
    auto alpha = 0.5f;
    auto beta  = 0.8f;
    p.add_instruction(migraphx::op::dot{alpha, beta}, t0, l1, t2);
    auto prog = migraphx::parse_onnx("gemm_test_ex1.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_vv)
{
    migraphx::program p;
    auto l0  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl0 = p.add_instruction(migraphx::op::unsqueeze{{0}}, l0);
    auto sl1 = p.add_instruction(migraphx::op::unsqueeze{{1}}, l1);
    auto res = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, sl0, sl1);
    auto sr0 = p.add_instruction(migraphx::op::squeeze{{0}}, res);
    p.add_instruction(migraphx::op::squeeze{{0}}, sr0);

    auto prog = migraphx::parse_onnx("matmul_vv.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_vm)
{
    migraphx::program p;
    auto l0  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7, 8}});
    auto sl0 = p.add_instruction(migraphx::op::unsqueeze{{0}}, l0);
    auto res = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, sl0, l1);
    p.add_instruction(migraphx::op::squeeze{{0}}, res);

    auto prog = migraphx::parse_onnx("matmul_vm.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_vbm)
{
    migraphx::program p;
    auto l0   = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1   = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {5, 7, 8}});
    auto sl0  = p.add_instruction(migraphx::op::unsqueeze{{0}}, l0);
    auto bsl0 = p.add_instruction(migraphx::op::multibroadcast{{5, 1, 7}}, sl0);
    std::cout << "ONNX_TEST" << std::endl;
    auto res = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, bsl0, l1);
    std::cout << "After Dot" << std::endl;
    p.add_instruction(migraphx::op::squeeze{{1}}, res);

    auto prog = migraphx::parse_onnx("matmul_vbm.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_mv)
{
    migraphx::program p;
    auto l0  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {6, 7}});
    auto l1  = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl1 = p.add_instruction(migraphx::op::unsqueeze{{1}}, l1);
    auto res = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, l0, sl1);
    p.add_instruction(migraphx::op::squeeze{{1}}, res);

    auto prog = migraphx::parse_onnx("matmul_mv.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_bmv)
{
    migraphx::program p;
    auto l0   = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1   = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl1  = p.add_instruction(migraphx::op::unsqueeze{{1}}, l1);
    auto bsl1 = p.add_instruction(migraphx::op::multibroadcast{{3, 7, 1}}, sl1);
    auto res  = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, l0, bsl1);
    p.add_instruction(migraphx::op::squeeze{{2}}, res);

    auto prog = migraphx::parse_onnx("matmul_bmv.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_bmbm)
{
    migraphx::program p;
    auto l0  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1  = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {5, 2, 1, 7, 8}});
    auto bl0 = p.add_instruction(migraphx::op::multibroadcast{{5, 2, 3, 6, 7}}, l0);
    auto bl1 = p.add_instruction(migraphx::op::multibroadcast{{5, 2, 3, 7, 8}}, l1);
    p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, bl0, bl1);

    auto prog = migraphx::parse_onnx("matmul_bmbm.onnx");

    EXPECT(p == prog);
}

TEST_CASE(add_scalar_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1}});
    auto m0 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l0);
    auto m1 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l1);
    p.add_instruction(migraphx::op::add{}, m0, m1);
    auto prog = migraphx::parse_onnx("add_scalar_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sub_scalar_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 =
        p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1}});
    auto m0 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l0);
    auto m1 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l1);
    p.add_instruction(migraphx::op::sub{}, m0, m1);
    auto prog = migraphx::parse_onnx("sub_scalar_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(group_conv_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 4, 16, 16}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 1, 3, 3}});
    migraphx::op::convolution op;
    op.group = 4;
    p.add_instruction(op, l0, l1);
    auto prog = migraphx::parse_onnx("group_conv_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pad_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    p.add_instruction(migraphx::op::pad{{1, 1, 1, 1}}, l0);
    auto prog = migraphx::parse_onnx("pad_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(lrn_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 28, 24, 24}});
    migraphx::op::lrn op;
    op.size  = 5;
    op.alpha = 0.0001;
    op.beta  = 0.75;
    op.bias  = 1.0;
    p.add_instruction(op, l0);
    auto prog = migraphx::parse_onnx("lrn_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(add_fp16_test)
{
    migraphx::program p;
    auto l0 =
        p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {1.5}});
    auto l1 =
        p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {2.5}});
    p.add_instruction(migraphx::op::add{}, l0, l1);
    auto prog = migraphx::parse_onnx("add_fp16_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(logsoftmax)
{
    migraphx::program p;
    auto l0  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    int axis = 1;
    p.add_instruction(migraphx::op::logsoftmax{axis}, l0);
    auto prog = migraphx::parse_onnx("logsoftmax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(argmax)
{
    migraphx::program p;
    auto l0  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto ins = p.add_instruction(migraphx::op::argmax{2}, l0);
    p.add_instruction(migraphx::op::squeeze{{2}}, ins);
    auto prog = migraphx::parse_onnx("argmax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(argmin)
{
    migraphx::program p;
    auto l0  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto ins = p.add_instruction(migraphx::op::argmin{3}, l0);
    p.add_instruction(migraphx::op::squeeze{{3}}, ins);
    auto prog = migraphx::parse_onnx("argmin_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(no_pad_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    p.add_instruction(migraphx::op::identity{}, l0);
    auto prog = migraphx::parse_onnx("no_pad_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_test1)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 = p.add_instruction(migraphx::op::reduce_sum{{2}}, l0);
    p.add_instruction(migraphx::op::squeeze{{2}}, l1);
    auto prog = migraphx::parse_onnx("reducesum_test1.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_test2)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 = p.add_instruction(migraphx::op::reduce_sum{{2, 3}}, l0);
    p.add_instruction(migraphx::op::squeeze{{2, 3}}, l1);
    auto prog = migraphx::parse_onnx("reducesum_test2.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_test3)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    p.add_instruction(migraphx::op::reduce_sum{{2, 3}}, l0);
    auto prog = migraphx::parse_onnx("reducesum_test3.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::clip{6.0, 0.0}, l0);
    auto prog = migraphx::parse_onnx("clip_test.onnx");

    EXPECT(p == prog);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
