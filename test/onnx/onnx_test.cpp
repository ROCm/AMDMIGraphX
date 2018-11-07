#include <iostream>
#include <vector>
#include <migraph/literal.hpp>
#include <migraph/operators.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/onnx.hpp>
#include "test.hpp"

void pytorch_conv_bias_test()
{
    migraph::program p;
    auto l0       = p.add_parameter("0", {migraph::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraph::shape::float_type, {1, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraph::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraph::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraph::op::broadcast{axis, l3->get_shape()}, l2);
    p.add_instruction(migraph::op::add{}, l3, l4);

    auto prog = migraph::parse_onnx("conv.onnx");
    EXPECT(p == prog);
}

void pytorch_conv_relu_maxpool()
{
    migraph::program p;
    auto l0       = p.add_parameter("0", {migraph::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraph::shape::float_type, {1, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraph::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraph::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraph::op::broadcast{axis, l3->get_shape()}, l2);
    auto l5       = p.add_instruction(migraph::op::add{}, l3, l4);
    auto l6       = p.add_instruction(migraph::op::relu{}, l5);
    p.add_instruction(migraph::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l6);

    auto prog = migraph::parse_onnx("conv_relu_maxpool.onnx");
    EXPECT(p == prog);
}

void pytorch_conv_bn_relu_maxpool()
{
    migraph::program p;
    auto l0 = p.add_parameter("0", {migraph::shape::float_type, {1, 3, 32, 32}});
    auto l1 = p.add_parameter("1", {migraph::shape::float_type, {1, 3, 5, 5}});
    auto l2 = p.add_parameter("2", {migraph::shape::float_type, {1}});

    auto p3       = p.add_parameter("3", {migraph::shape::float_type, {1}});
    auto p4       = p.add_parameter("4", {migraph::shape::float_type, {1}});
    auto p5       = p.add_parameter("5", {migraph::shape::float_type, {1}});
    auto p6       = p.add_parameter("6", {migraph::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraph::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraph::op::broadcast{axis, l3->get_shape()}, l2);
    auto l5       = p.add_instruction(migraph::op::add{}, l3, l4);
    auto l6 = p.add_instruction(migraph::op::batch_norm_inference{1.0e-5f}, l5, p3, p4, p5, p6);
    auto l7 = p.add_instruction(migraph::op::relu{}, l6);
    p.add_instruction(migraph::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l7);

    auto prog = migraph::parse_onnx("conv_bn_relu_maxpool.onnx");
    EXPECT(p == prog);
}

void pytorch_conv_relu_maxpool_x2()
{
    migraph::program p;
    auto l0       = p.add_parameter("0", {migraph::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraph::shape::float_type, {5, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraph::shape::float_type, {5}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraph::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraph::op::broadcast{axis, l3->get_shape()}, l2);
    auto l5       = p.add_instruction(migraph::op::add{}, l3, l4);
    auto l6       = p.add_instruction(migraph::op::relu{}, l5);
    auto l7 = p.add_instruction(migraph::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l6);

    auto l8  = p.add_parameter("3", {migraph::shape::float_type, {1, 5, 5, 5}});
    auto l9  = p.add_parameter("4", {migraph::shape::float_type, {1}});
    auto l10 = p.add_instruction(migraph::op::convolution{}, l7, l8);
    auto l11 = p.add_instruction(migraph::op::broadcast{axis, l10->get_shape()}, l9);
    auto l12 = p.add_instruction(migraph::op::add{}, l10, l11);
    auto l13 = p.add_instruction(migraph::op::relu{}, l12);
    p.add_instruction(migraph::op::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l13);

    auto prog = migraph::parse_onnx("conv_relu_maxpoolX2.onnx");

    EXPECT(p == prog);
}

void leaky_relu_test()
{
    migraph::program p;
    float alpha = 0.01f;
    auto l0     = p.add_parameter("0", {migraph::shape::float_type, {3}});
    p.add_instruction(migraph::op::leaky_relu{alpha}, l0);

    auto prog = migraph::parse_onnx("leaky_relu.onnx");

    EXPECT(p == prog);
}

void imagescaler_test()
{
    migraph::program p;
    migraph::shape s{migraph::shape::float_type, {1, 3, 16, 16}};
    auto l0        = p.add_parameter("0", s);
    auto scale_val = p.add_literal(0.5f);
    auto bias_vals = p.add_literal(
        migraph::literal{migraph::shape{migraph::shape::float_type, {3}}, {0.01, 0.02, 0.03}});
    auto scaled_tensor = p.add_instruction(migraph::op::scalar{s}, scale_val);
    auto img_scaled    = p.add_instruction(migraph::op::mul{}, l0, scaled_tensor);
    auto bias_bcast    = p.add_instruction(migraph::op::broadcast{1, s}, bias_vals);
    p.add_instruction(migraph::op::add{}, img_scaled, bias_bcast);

    auto prog = migraph::parse_onnx("imagescaler_test.onnx");

    EXPECT(p == prog);
}

void globalavgpool_test()
{
    migraph::program p;
    auto input = p.add_parameter("0", migraph::shape{migraph::shape::float_type, {1, 3, 16, 16}});
    auto op    = migraph::op::pooling{"average"};
    auto lens  = input->get_shape().lens();
    op.lengths = {lens[2], lens[3]};
    p.add_instruction(op, input);

    auto prog = migraph::parse_onnx("globalavgpool_test.onnx");

    EXPECT(p == prog);
}

void globalmaxpool_test()
{
    migraph::program p;
    auto input = p.add_parameter("0", migraph::shape{migraph::shape::float_type, {1, 3, 16, 16}});
    auto op    = migraph::op::pooling{"max"};
    auto lens  = input->get_shape().lens();
    op.lengths = {lens[2], lens[3]};
    p.add_instruction(op, input);

    auto prog = migraph::parse_onnx("globalmaxpool_test.onnx");

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
}
