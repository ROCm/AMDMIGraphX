#include <iostream>
#include <vector>
#include <migraph/literal.hpp>
#include <migraph/operators.hpp>
#include <migraph/program.hpp>
#include <migraph/onnx.hpp>
#include "test.hpp"
#include "verify.hpp"

void pytorch_conv_bias_test()
{
    migraph::program p;
    auto l0       = p.add_parameter("0", {migraph::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraph::shape::float_type, {1, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraph::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraph::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraph::broadcast{axis}, l3, l2);
    p.add_instruction(migraph::add{}, l3, l4);

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
    auto l3       = p.add_instruction(migraph::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraph::broadcast{axis}, l3, l2);
    auto l5       = p.add_instruction(migraph::add{}, l3, l4);
    auto l6       = p.add_instruction(migraph::activation{"relu"}, l5);
    p.add_instruction(migraph::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l6);

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
    auto l3       = p.add_instruction(migraph::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraph::broadcast{axis}, l3, l2);
    auto l5       = p.add_instruction(migraph::add{}, l3, l4);
    auto l6       = p.add_instruction(migraph::batch_norm_inference{}, l5, p3, p4, p5, p6);
    auto l7       = p.add_instruction(migraph::activation{"relu"}, l6);
    p.add_instruction(migraph::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l7);

    auto prog = migraph::parse_onnx("conv_bn_relu_maxpool.onnx");
    EXPECT(p == prog);
}

void pytorch_conv_relu_maxpoolX2()
{
    migraph::program p;
    auto l0       = p.add_parameter("0", {migraph::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraph::shape::float_type, {5, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraph::shape::float_type, {5}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraph::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraph::broadcast{axis}, l3, l2);
    auto l5       = p.add_instruction(migraph::add{}, l3, l4);
    auto l6       = p.add_instruction(migraph::activation{"relu"}, l5);
    auto l7       = p.add_instruction(migraph::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l6);

    auto l8  = p.add_parameter("3", {migraph::shape::float_type, {1, 5, 5, 5}});
    auto l9  = p.add_parameter("4", {migraph::shape::float_type, {1}});
    auto l10 = p.add_instruction(migraph::convolution{}, l7, l8);
    auto l11 = p.add_instruction(migraph::broadcast{axis}, l10, l9);
    auto l12 = p.add_instruction(migraph::add{}, l10, l11);
    auto l13 = p.add_instruction(migraph::activation{"relu"}, l12);
    p.add_instruction(migraph::pooling{"max", {{0, 0}}, {{2, 2}}, {{2, 2}}}, l13);

    auto prog = migraph::parse_onnx("conv_relu_maxpoolX2.onnx");

    EXPECT(p == prog);
}

int main()
{
    pytorch_conv_bias_test();
    pytorch_conv_relu_maxpool();
    pytorch_conv_bn_relu_maxpool();
    pytorch_conv_relu_maxpoolX2();
}
