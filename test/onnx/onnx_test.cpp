#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

void pytorch_conv_bias_test()
{
    migraphx::program p;
    auto l0       = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape()}, l2);
    p.add_instruction(migraphx::op::add{}, l3, l4);

    auto prog = migraphx::parse_onnx("conv.onnx");
    EXPECT(p == prog);
}

void pytorch_conv_relu_maxpool()
{
    migraphx::program p;
    auto l0       = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape()}, l2);
    auto l5       = p.add_instruction(migraphx::op::add{}, l3, l4);
    auto l6       = p.add_instruction(migraphx::op::relu{}, l5);
    p.add_instruction(migraphx::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l6);

    auto prog = migraphx::parse_onnx("conv_relu_maxpool.onnx");
    EXPECT(p == prog);
}

void pytorch_conv_bn_relu_maxpool()
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
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape()}, l2);
    auto l5       = p.add_instruction(migraphx::op::add{}, l3, l4);
    auto l6 = p.add_instruction(migraphx::op::batch_norm_inference{1.0e-5f}, l5, p3, p4, p5, p6);
    auto l7 = p.add_instruction(migraphx::op::relu{}, l6);
    p.add_instruction(migraphx::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l7);

    auto prog = migraphx::parse_onnx("conv_bn_relu_maxpool.onnx");
    EXPECT(p == prog);
}

void pytorch_conv_relu_maxpool_x2()
{
    migraphx::program p;
    auto l0       = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraphx::shape::float_type, {5, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraphx::shape::float_type, {5}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape()}, l2);
    auto l5       = p.add_instruction(migraphx::op::add{}, l3, l4);
    auto l6       = p.add_instruction(migraphx::op::relu{}, l5);
    auto l7 = p.add_instruction(migraphx::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l6);

    auto l8  = p.add_parameter("3", {migraphx::shape::float_type, {1, 5, 5, 5}});
    auto l9  = p.add_parameter("4", {migraphx::shape::float_type, {1}});
    auto l10 = p.add_instruction(migraphx::op::convolution{}, l7, l8);
    auto l11 = p.add_instruction(migraphx::op::broadcast{axis, l10->get_shape()}, l9);
    auto l12 = p.add_instruction(migraphx::op::add{}, l10, l11);
    auto l13 = p.add_instruction(migraphx::op::relu{}, l12);
    p.add_instruction(migraphx::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l13);

    auto prog = migraphx::parse_onnx("conv_relu_maxpoolX2.onnx");

    EXPECT(p == prog);
}

void leaky_relu_test()
{
    migraphx::program p;
    float alpha = 0.01f;
    auto l0     = p.add_parameter("0", {migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::leaky_relu{alpha}, l0);

    auto prog = migraphx::parse_onnx("leaky_relu.onnx");

    EXPECT(p == prog);
}

void imagescaler_test()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 16, 16}};
    auto l0        = p.add_parameter("0", s);
    auto scale_val = p.add_literal(0.5f);
    auto bias_vals = p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0.01, 0.02, 0.03}});
    auto scaled_tensor = p.add_instruction(migraphx::op::scalar{s}, scale_val);
    auto img_scaled    = p.add_instruction(migraphx::op::mul{}, l0, scaled_tensor);
    auto bias_bcast    = p.add_instruction(migraphx::op::broadcast{1, s}, bias_vals);
    p.add_instruction(migraphx::op::add{}, img_scaled, bias_bcast);

    auto prog = migraphx::parse_onnx("imagescaler_test.onnx");

    EXPECT(p == prog);
}

void globalavgpool_test()
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

void globalmaxpool_test()
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

void transpose_test()
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    std::vector<int64_t> perm{0, 3, 1, 2};
    p.add_instruction(migraphx::op::transpose{perm}, input);

    auto prog = migraphx::parse_onnx("transpose_test.onnx");

    EXPECT(p == prog);
}

void dropout_test()
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}});
    p.add_instruction(migraphx::op::identity{}, input);

    auto prog = migraphx::parse_onnx("dropout_test.onnx");

    EXPECT(p == prog);
}

void sum_test()
{
    migraphx::program p;
    auto input0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = p.add_instruction(migraphx::op::add{}, input0, input1);
    p.add_instruction(migraphx::op::add{}, l0, input2);

    auto prog = migraphx::parse_onnx("sum_test.onnx");
}

void sin_test()
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::sin{}, input);

    auto prog = migraphx::parse_onnx("sin_test.onnx");
    EXPECT(p == prog);
}

void cos_test()
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::cos{}, input);

    auto prog = migraphx::parse_onnx("cos_test.onnx");
    EXPECT(p == prog);
}

void tan_test()
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::tan{}, input);

    auto prog = migraphx::parse_onnx("tan_test.onnx");
    EXPECT(p == prog);
}

void sinh_test()
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::sinh{}, input);

    auto prog = migraphx::parse_onnx("sinh_test.onnx");

    EXPECT(p == prog);
}

void cosh_test()
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    p.add_instruction(migraphx::op::cosh{}, input);

    auto prog = migraphx::parse_onnx("cosh_test.onnx");

    EXPECT(p == prog);
}

void tanh_test()
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    p.add_instruction(migraphx::op::tanh{}, input);

    auto prog = migraphx::parse_onnx("tanh_test.onnx");

    EXPECT(p == prog);
}

void asin_test()
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::asin{}, input);

    auto prog = migraphx::parse_onnx("asin_test.onnx");

    EXPECT(p == prog);
}

void max_test()
{
    migraphx::program p;
    auto input0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = p.add_instruction(migraphx::op::max{}, input0, input1);
    p.add_instruction(migraphx::op::max{}, l0, input2);

    auto prog = migraphx::parse_onnx("max_test.onnx");
}

void acos_test()
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::acos{}, input);

    auto prog = migraphx::parse_onnx("acos_test.onnx");

    EXPECT(p == prog);
}

void min_test()
{
    migraphx::program p;
    auto input0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = p.add_instruction(migraphx::op::min{}, input0, input1);
    p.add_instruction(migraphx::op::min{}, l0, input2);

    auto prog = migraphx::parse_onnx("min_test.onnx");
}

void atan_test()
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::atan{}, input);

    auto prog = migraphx::parse_onnx("atan_test.onnx");

    EXPECT(p == prog);
}

void add_bcast_test()
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2 = p.add_instruction(migraphx::op::broadcast{1, l0->get_shape()}, l1);
    p.add_instruction(migraphx::op::add{}, l0, l2);

    auto prog = migraphx::parse_onnx("add_bcast_test.onnx");

    EXPECT(p == prog);
}

void implicit_bcast_test()
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2 = p.add_instruction(migraphx::op::multibroadcast{{0, 0, 4, 5}}, l0);
    auto l3 = p.add_instruction(migraphx::op::multibroadcast{{0, 0, 4, 5}}, l1);
    p.add_instruction(migraphx::op::add{}, l2, l3);

    auto prog = migraphx::parse_onnx("implicit_bcast_test.onnx");

    EXPECT(p == prog);
}

void unknown_test()
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    p.add_instruction(migraphx::unknown{"Unknown"}, l0, l1);
    p.add_instruction(migraphx::unknown{"Unknown"});
    auto prog = migraphx::parse_onnx("unknown_test.onnx");

    EXPECT(p == prog);
}

void softmax_test()
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    p.add_instruction(migraphx::op::softmax{}, l0);
    auto prog = migraphx::parse_onnx("softmax_test.onnx");

    EXPECT(p == prog);
}

void reshape_test()
{
    migraphx::program p;
    migraphx::op::reshape op;
    std::vector<int64_t> reshape_dims{3, 8};
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});
    p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, reshape_dims});
    op.dims = reshape_dims;
    p.add_instruction(op, l0);
    p.add_instruction(op, l0);
    auto prog = migraphx::parse_onnx("reshape_test.onnx");

    EXPECT(p == prog);
}

int main()
{
    pytorch_conv_bias_test();
    pytorch_conv_relu_maxpool();
    pytorch_conv_bn_relu_maxpool();
    pytorch_conv_relu_maxpool_x2();
    leaky_relu_test();
    imagescaler_test();
    globalavgpool_test();
    globalmaxpool_test();
    transpose_test();
    dropout_test();
    sum_test();
    max_test();
    min_test();
    sin_test();
    cos_test();
    tan_test();
    sinh_test();
    cosh_test();
    tanh_test();
    asin_test();
    acos_test();
    atan_test();
    add_bcast_test();
    implicit_bcast_test();
    unknown_test();
    reshape_test();
}
