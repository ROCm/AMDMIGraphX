#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

migraphx::program optimize_onnx(const std::string& name, bool eliminate_deadcode = false)
{
    migraphx::onnx_options options;
    options.skip_unknown_operators = true;
    auto prog                      = migraphx::parse_onnx(name, options);
    if(eliminate_deadcode)
        migraphx::run_passes(prog, {migraphx::dead_code_elimination{}});

    // remove the last identity instruction
    auto last_ins = std::prev(prog.end());
    if(last_ins->name() == "@return")
    {
        prog.remove_instruction(last_ins);
    }

    return prog;
}

TEST_CASE(acos_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::acos{}, input);

    auto prog = optimize_onnx("acos_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(acosh_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::acosh{}, input);

    auto prog = optimize_onnx("acosh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(add_bcast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2 = p.add_instruction(migraphx::op::broadcast{1, l0->get_shape().lens()}, l1);
    p.add_instruction(migraphx::op::add{}, l0, l2);

    auto prog = optimize_onnx("add_bcast_test.onnx");

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
    auto prog = optimize_onnx("add_fp16_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(add_scalar_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::uint8_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::uint8_type});
    auto m1 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l1);
    auto r  = p.add_instruction(migraphx::op::add{}, l0, m1);
    p.add_return({r});
    auto prog = migraphx::parse_onnx("add_scalar_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(argmax_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto ins = p.add_instruction(migraphx::op::argmax{2}, l0);
    p.add_instruction(migraphx::op::squeeze{{2}}, ins);
    auto prog = optimize_onnx("argmax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(argmin_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto ins = p.add_instruction(migraphx::op::argmin{3}, l0);
    p.add_instruction(migraphx::op::squeeze{{3}}, ins);
    auto prog = optimize_onnx("argmin_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(asin_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::asin{}, input);

    auto prog = optimize_onnx("asin_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(asinh_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::asinh{}, input);

    auto prog = optimize_onnx("asinh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(atan_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::atan{}, input);

    auto prog = optimize_onnx("atan_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(atanh_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::atanh{}, input);

    auto prog = optimize_onnx("atanh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(averagepool_1d_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 5}});
    p.add_instruction(migraphx::op::pooling{"average", {0}, {1}, {3}}, l0);

    auto prog = optimize_onnx("averagepool_1d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(averagepool_3d_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 5, 5, 5}});
    p.add_instruction(migraphx::op::pooling{"average", {0, 0, 0}, {1, 1, 1}, {3, 3, 3}}, l0);

    auto prog = optimize_onnx("averagepool_3d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(averagepool_notset_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    auto ins   = p.add_instruction(migraphx::op::pooling{"average", {2, 2}, {2, 2}, {6, 6}}, input);
    auto ret   = p.add_instruction(migraphx::op::slice{{2, 3}, {1, 1}, {2, 2}}, ins);
    p.add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_notset_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(averagepool_nt_cip_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0, 1, 1};
    auto ins_pad              = p.add_instruction(migraphx::op::pad{pads}, input);
    auto ret = p.add_instruction(migraphx::op::pooling{"average", {0, 0}, {2, 2}, {6, 6}}, ins_pad);
    p.add_return({ret});

    auto prog = migraphx::parse_onnx("averagepool_nt_cip_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(averagepool_same_lower_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    auto ins   = p.add_instruction(
        migraphx::op::pooling{
            "average", {1, 1}, {1, 1}, {2, 2}, migraphx::op::padding_mode_t::same},
        input);
    auto ret = p.add_instruction(migraphx::op::slice{{2, 3}, {0, 0}, {5, 5}}, ins);
    p.add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_same_lower_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(averagepool_sl_cip_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    std::vector<int64_t> pads = {0, 0, 1, 1, 0, 0, 0, 0};
    auto ins_pad              = p.add_instruction(migraphx::op::pad{pads}, input);
    auto ret                  = p.add_instruction(
        migraphx::op::pooling{
            "average", {0, 0}, {1, 1}, {2, 2}, migraphx::op::padding_mode_t::same},
        ins_pad);
    p.add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_sl_cip_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(averagepool_same_upper_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    auto ins   = p.add_instruction(
        migraphx::op::pooling{
            "average", {1, 1}, {1, 1}, {2, 2}, migraphx::op::padding_mode_t::same},
        input);
    auto ret = p.add_instruction(migraphx::op::slice{{2, 3}, {1, 1}, {6, 6}}, ins);
    p.add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_same_upper_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(batchnorm_1d_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 5}});
    auto l1 = p.add_parameter("1", {migraphx::shape::float_type, {3}});
    auto l2 = p.add_parameter("2", {migraphx::shape::float_type, {3}});
    auto l3 = p.add_parameter("3", {migraphx::shape::float_type, {3}});
    auto l4 = p.add_parameter("4", {migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::batch_norm_inference{}, l0, l1, l2, l3, l4);

    auto prog = optimize_onnx("batchnorm_1d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(batchnorm_3d_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 5, 5, 5}});
    auto l1 = p.add_parameter("1", {migraphx::shape::float_type, {3}});
    auto l2 = p.add_parameter("2", {migraphx::shape::float_type, {3}});
    auto l3 = p.add_parameter("3", {migraphx::shape::float_type, {3}});
    auto l4 = p.add_parameter("4", {migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::batch_norm_inference{}, l0, l1, l2, l3, l4);

    auto prog = optimize_onnx("batchnorm_3d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(cast_test)
{
    migraphx::program p;
    auto l = p.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {10}});
    p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, l);

    auto prog = optimize_onnx("cast_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(ceil_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::ceil{}, input);

    auto prog = optimize_onnx("ceil_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test)
{
    migraphx::program p;
    auto l0      = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto min_val = p.add_literal(0.0f);
    auto max_val = p.add_literal(6.0f);
    min_val      = p.add_instruction(migraphx::op::multibroadcast{{3}}, min_val);
    max_val      = p.add_instruction(migraphx::op::multibroadcast{{3}}, max_val);
    p.add_instruction(migraphx::op::clip{}, l0, min_val, max_val);
    auto prog = optimize_onnx("clip_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test_op11)
{
    migraphx::program p;
    auto min_val = p.add_literal(0.0f);
    auto max_val = p.add_literal(6.0f);
    auto l0      = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    min_val      = p.add_instruction(migraphx::op::multibroadcast{{3}}, min_val);
    max_val      = p.add_instruction(migraphx::op::multibroadcast{{3}}, max_val);
    p.add_instruction(migraphx::op::clip{}, l0, min_val, max_val);
    auto prog = optimize_onnx("clip_test_op11.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test_op11_min_only)
{
    migraphx::program p;
    auto min_val = p.add_literal(0.0f);
    auto l0      = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    min_val      = p.add_instruction(migraphx::op::multibroadcast{{3}}, min_val);
    p.add_instruction(migraphx::op::max{}, l0, min_val);
    auto prog = optimize_onnx("clip_test_op11_min_only.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test_op11_no_args)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::identity{}, l0);
    auto prog = optimize_onnx("clip_test_op11_no_args.onnx");

    EXPECT(p == prog);
}

TEST_CASE(concat_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7, 4, 3}});
    p.add_instruction(migraphx::op::concat{0}, l0, l1);
    auto prog = optimize_onnx("concat_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_test)
{
    migraphx::program p;
    p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0, 1, 2}});
    auto prog = optimize_onnx("constant_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_fill_test)
{

    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> value(s.elements(), 1.0);
    p.add_literal(migraphx::literal{s, value});
    auto prog = optimize_onnx("constant_fill_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_fill_input_as_shape_test)
{
    migraphx::program p;
    auto l0 = p.add_literal(migraphx::literal{{migraphx::shape::int32_type, {2}}, {2, 3}});
    std::vector<std::size_t> dims(l0->get_shape().elements());
    migraphx::literal ls = l0->get_literal();
    ls.visit([&](auto s) { dims.assign(s.begin(), s.end()); });
    migraphx::shape s{migraphx::shape::float_type, dims};
    std::vector<float> value(s.elements(), 1.0);
    p.add_literal(migraphx::literal{s, value});
    auto prog = optimize_onnx("constant_fill_input_as_shape_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_scalar_test)
{
    migraphx::program p;
    p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1}}, {1}});
    auto prog = optimize_onnx("constant_scalar_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(const_of_shape_empty_input_test)
{
    migraphx::program p;
    p.add_literal(migraphx::literal());
    migraphx::shape s(migraphx::shape::int64_type, {1}, {0});
    std::vector<int64_t> vec(s.elements(), 10);
    p.add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_empty_input_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(const_of_shape_float_test)
{
    migraphx::program p;
    migraphx::shape ss(migraphx::shape::int32_type, {3});
    p.add_literal(migraphx::literal(ss, {2, 3, 4}));
    migraphx::shape s(migraphx::shape::float_type, {2, 3, 4});
    std::vector<float> vec(s.elements(), 10.0f);
    p.add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_float_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(const_of_shape_int64_test)
{
    migraphx::program p;
    migraphx::shape ss(migraphx::shape::int32_type, {3});
    p.add_literal(migraphx::literal(ss, {2, 3, 4}));
    migraphx::shape s(migraphx::shape::int64_type, {2, 3, 4});
    std::vector<int64_t> vec(s.elements(), 10);
    p.add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_int64_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(const_of_shape_no_value_attr_test)
{
    migraphx::program p;
    migraphx::shape ss(migraphx::shape::int32_type, {3});
    p.add_literal(migraphx::literal(ss, {2, 3, 4}));
    migraphx::shape s(migraphx::shape::float_type, {2, 3, 4});
    std::vector<float> vec(s.elements(), 0.0f);
    p.add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_no_value_attr_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_autopad_fail_test)
{
    EXPECT(test::throws([&] { optimize_onnx("conv_autopad_fail_test.onnx"); }));
}

TEST_CASE(conv_1d_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 5}});
    auto l1 = p.add_parameter("1", {migraphx::shape::float_type, {1, 3, 3}});
    p.add_instruction(migraphx::op::convolution{{0}, {1}, {1}}, l0, l1);

    auto prog = optimize_onnx("conv_1d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_3d_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 5, 5, 5}});
    auto l1 = p.add_parameter("1", {migraphx::shape::float_type, {1, 3, 3, 3, 3}});
    p.add_instruction(migraphx::op::convolution{{0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, l0, l1);

    auto prog = optimize_onnx("conv_3d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_attr_fail_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("conv_attr_fail_test.onnx"); }));
}

TEST_CASE(conv_autopad_same_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1 = p.add_parameter("1", {migraphx::shape::float_type, {1, 3, 3, 3}});
    migraphx::op::convolution op;
    op.padding      = {1, 1};
    op.padding_mode = migraphx::op::padding_mode_t::same;
    p.add_instruction(op, l0, l1);

    auto prog = optimize_onnx("conv_autopad_same_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_bias_test)
{
    migraphx::program p;
    auto l0       = p.add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape().lens()}, l2);
    p.add_instruction(migraphx::op::add{}, l3, l4);

    auto prog = optimize_onnx("conv_bias_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_bn_relu_maxpool_test)
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

    auto prog = optimize_onnx("conv_bn_relu_maxpool_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_relu_maxpool_test)
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

    auto prog = optimize_onnx("conv_relu_maxpool_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_relu_maxpool_x2_test)
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

    auto prog = optimize_onnx("conv_relu_maxpool_x2_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(convinteger_bias_test)
{
    migraphx::program p;
    auto l0       = p.add_parameter("0", {migraphx::shape::int8_type, {1, 3, 32, 32}});
    auto l1       = p.add_parameter("1", {migraphx::shape::int8_type, {1, 3, 5, 5}});
    auto l2       = p.add_parameter("2", {migraphx::shape::int32_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::quant_convolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape().lens()}, l2);
    p.add_instruction(migraphx::op::add{}, l3, l4);

    auto prog = optimize_onnx("convinteger_bias_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(cos_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::cos{}, input);

    auto prog = optimize_onnx("cos_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(cosh_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    p.add_instruction(migraphx::op::cosh{}, input);

    auto prog = optimize_onnx("cosh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(deconv_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1 = p.add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});
    p.add_instruction(migraphx::op::deconvolution{}, l0, l1);

    auto prog = optimize_onnx("deconv_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_bias_test)
{
    migraphx::program p;
    auto l0       = p.add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1       = p.add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l2       = p.add_parameter("b", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = p.add_instruction(migraphx::op::deconvolution{}, l0, l1);
    auto l4       = p.add_instruction(migraphx::op::broadcast{axis, l3->get_shape().lens()}, l2);
    p.add_instruction(migraphx::op::add{}, l3, l4);

    auto prog = optimize_onnx("deconv_bias_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_input_pads_strides_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1 = p.add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    p.add_instruction(migraphx::op::deconvolution{{1, 1}, {3, 2}}, l0, l1);

    auto prog = optimize_onnx("deconv_input_pads_strides_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_input_pads_asymm_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1 = p.add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    auto l2 = p.add_instruction(migraphx::op::deconvolution{{0, 0}, {3, 2}}, l0, l1);
    p.add_instruction(migraphx::op::slice{{0, 1, 2, 3}, {0, 0, 0, 0}, {1, 2, 8, 6}}, l2);

    auto prog = optimize_onnx("deconv_input_pads_asymm_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_output_shape_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1 = p.add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    auto l2 = p.add_instruction(migraphx::op::deconvolution{{0, 0}, {3, 2}}, l0, l1);
    p.add_instruction(migraphx::op::pad{{0, 0, 0, 0, 0, 0, 1, 1}}, l2);

    auto prog = optimize_onnx("deconv_output_shape_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_output_padding_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1 = p.add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    auto l2 = p.add_instruction(migraphx::op::deconvolution{{0, 0}, {3, 2}}, l0, l1);
    p.add_instruction(migraphx::op::pad{{0, 0, 0, 0, 0, 0, 1, 1}}, l2);

    auto prog = optimize_onnx("deconv_output_padding_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(dropout_test)
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}});
    p.add_instruction(migraphx::op::identity{}, input);

    auto prog = optimize_onnx("dropout_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(elu_test)
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::elu{0.01}, input);

    auto prog = optimize_onnx("elu_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(embedding_bag_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("weight", migraphx::shape{migraphx::shape::float_type, {4, 2}});
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {3}}, {1, 0, 2}};
    auto l1 = p.add_literal(l);
    p.add_literal(0);
    auto l4 = p.add_instruction(migraphx::op::gather{}, l0, l1);
    auto r1 = p.add_instruction(migraphx::op::reduce_sum{{0}}, l4);
    auto l5 = p.add_instruction(migraphx::op::gather{}, l0, l1);
    auto r2 = p.add_instruction(migraphx::op::reduce_mean{{0}}, l5);
    auto l6 = p.add_instruction(migraphx::op::gather{}, l0, l1);
    auto r3 = p.add_instruction(migraphx::op::reduce_max{{0}}, l6);
    p.add_return({r1, r2, r3});

    auto prog = migraphx::parse_onnx("embedding_bag_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(embedding_bag_offset_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("embedding_bag_offset_test.onnx"); }));
}

TEST_CASE(erf_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    p.add_instruction(migraphx::op::erf{}, input);

    auto prog = optimize_onnx("erf_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(exp_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::exp{}, input);

    auto prog = optimize_onnx("exp_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(expand_test)
{
    migraphx::program p;
    migraphx::shape s(migraphx::shape::float_type, {3, 1, 1});
    auto param = p.add_parameter("x", s);
    migraphx::shape ss(migraphx::shape::int32_type, {4});
    p.add_literal(migraphx::literal(ss, {2, 3, 4, 5}));
    p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, param);

    auto prog = optimize_onnx("expand_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(flatten_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    p.add_instruction(migraphx::op::flatten{2}, l0);
    p.add_instruction(migraphx::op::flatten{1}, l0);
    auto prog = optimize_onnx("flatten_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(floor_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::floor{}, input);

    auto prog = optimize_onnx("floor_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gather_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = p.add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, {2, 3}});
    int axis = 1;
    p.add_instruction(migraphx::op::gather{axis}, l0, l1);
    auto prog = optimize_onnx("gather_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gather_elements_axis0_test)
{
    migraphx::program p;
    auto data    = p.add_parameter("data", {migraphx::shape::float_type, {3, 4}});
    auto indices = p.add_parameter("indices", {migraphx::shape::int32_type, {2, 3}});
    std::vector<int> ind_indices{0, 1, 2, 4, 5, 6};
    std::vector<int> ind_axis_indices{0, 0, 0, 1, 1, 1};
    migraphx::shape ind_s{migraphx::shape::int32_type, {2, 3}};
    auto l_data_indices =
        p.add_literal(migraphx::literal{ind_s, ind_indices.begin(), ind_indices.end()});
    auto l_ind_axis_indices =
        p.add_literal(migraphx::literal{ind_s, ind_axis_indices.begin(), ind_axis_indices.end()});
    auto l_stride = p.add_literal(migraphx::literal{{migraphx::shape::int32_type, {1}}, {4}});

    auto rsp_data    = p.add_instruction(migraphx::op::reshape{{12}}, data);
    auto lbst_stride = p.add_instruction(migraphx::op::multibroadcast{ind_s.lens()}, l_stride);
    auto axis_delta  = p.add_instruction(migraphx::op::sub{}, indices, l_ind_axis_indices);
    auto mul_delta   = p.add_instruction(migraphx::op::mul{}, axis_delta, lbst_stride);
    auto ind         = p.add_instruction(migraphx::op::add{}, l_data_indices, mul_delta);
    auto ret         = p.add_instruction(migraphx::op::gather{0}, rsp_data, ind);
    p.add_return({ret});

    auto prog = migraphx::parse_onnx("gather_elements_axis0_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gather_elements_axis1_test)
{
    migraphx::program p;
    auto data    = p.add_parameter("data", {migraphx::shape::float_type, {3, 4}});
    auto indices = p.add_parameter("indices", {migraphx::shape::int32_type, {2, 3}});
    std::vector<int> ind_indices{0, 1, 2, 4, 5, 6};
    std::vector<int> ind_axis_indices{0, 1, 2, 0, 1, 2};
    migraphx::shape ind_s{migraphx::shape::int32_type, {2, 3}};
    auto l_data_indices =
        p.add_literal(migraphx::literal{ind_s, ind_indices.begin(), ind_indices.end()});
    auto l_ind_axis_indices =
        p.add_literal(migraphx::literal{ind_s, ind_axis_indices.begin(), ind_axis_indices.end()});
    auto l_stride = p.add_literal(migraphx::literal{{migraphx::shape::int32_type, {1}}, {1}});

    auto rsp_data    = p.add_instruction(migraphx::op::reshape{{12}}, data);
    auto lbst_stride = p.add_instruction(migraphx::op::multibroadcast{ind_s.lens()}, l_stride);
    auto axis_delta  = p.add_instruction(migraphx::op::sub{}, indices, l_ind_axis_indices);
    auto mul_delta   = p.add_instruction(migraphx::op::mul{}, axis_delta, lbst_stride);
    auto ind         = p.add_instruction(migraphx::op::add{}, l_data_indices, mul_delta);
    auto ret         = p.add_instruction(migraphx::op::gather{0}, rsp_data, ind);
    p.add_return({ret});

    auto prog = migraphx::parse_onnx("gather_elements_axis1_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gemm_test)
{
    migraphx::program p;
    auto l0    = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 7}});
    auto l1    = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {11, 5}});
    auto l2    = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type});
    auto t0    = p.add_instruction(migraphx::op::transpose{{1, 0}}, l0);
    auto t1    = p.add_instruction(migraphx::op::transpose{{1, 0}}, l1);
    auto bl2   = p.add_instruction(migraphx::op::multibroadcast{{7, 11}}, l2);
    auto alpha = 2.f;
    auto beta  = 2.0f;
    p.add_instruction(migraphx::op::dot{alpha, beta}, t0, t1, bl2);
    auto prog = optimize_onnx("gemm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gemm_ex_test)
{
    migraphx::program p;
    auto l0    = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 6}});
    auto l1    = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 7}});
    auto l2    = p.add_parameter("3", migraphx::shape{migraphx::shape::float_type, {1, 1, 6, 7}});
    auto t0    = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, l0);
    auto alpha = 0.5f;
    auto beta  = 0.8f;
    p.add_instruction(migraphx::op::dot{alpha, beta}, t0, l1, l2);
    auto prog = optimize_onnx("gemm_ex_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gemm_ex_brcst_test)
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
    auto prog = optimize_onnx("gemm_ex_brcst_test.onnx");

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

    auto prog = optimize_onnx("globalavgpool_test.onnx");

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

    auto prog = optimize_onnx("globalmaxpool_test.onnx");

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
    auto prog = optimize_onnx("group_conv_test.onnx");

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

    auto prog = optimize_onnx("imagescaler_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(imagescaler_half_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::half_type, {1, 3, 16, 16}};
    auto l0 = p.add_parameter("0", s);
    auto scale_val =
        p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {0.5f}});
    auto bias_vals = p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::half_type, {3}}, {0.01, 0.02, 0.03}});
    auto scaled_tensor = p.add_instruction(migraphx::op::scalar{s.lens()}, scale_val);
    auto img_scaled    = p.add_instruction(migraphx::op::mul{}, l0, scaled_tensor);
    auto bias_bcast    = p.add_instruction(migraphx::op::broadcast{1, s.lens()}, bias_vals);
    p.add_instruction(migraphx::op::add{}, img_scaled, bias_bcast);

    auto prog = optimize_onnx("imagescaler_half_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(implicit_add_bcast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4, 1}});
    auto l3 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l1);
    p.add_instruction(migraphx::op::add{}, l0, l3);

    auto prog = optimize_onnx("implicit_add_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(implicit_add_bcast_user_input_shape_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 5, 1}});
    auto l3 = p.add_instruction(migraphx::op::multibroadcast{{3, 4, 5, 6}}, l1);
    auto r  = p.add_instruction(migraphx::op::add{}, l0, l3);
    p.add_return({r});

    migraphx::onnx_options options;
    options.map_input_dims["0"] = {3, 4, 5, 6};
    options.map_input_dims["1"] = {4, 5, 1};
    auto prog                   = migraphx::parse_onnx("implicit_add_bcast_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(implicit_pow_bcast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4, 1}});
    auto l3 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l1);
    p.add_instruction(migraphx::op::pow{}, l0, l3);

    auto prog = optimize_onnx("implicit_pow_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(implicit_sub_bcast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 5}});
    auto l3 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l1);
    p.add_instruction(migraphx::op::sub{}, l0, l3);

    auto prog = optimize_onnx("implicit_sub_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(initializer_not_an_input)
{
    migraphx::program p;
    std::vector<float> w = {1, 2, 3, 4, 5, 6, 7, 8};
    auto l1 = p.add_literal(migraphx::literal({migraphx::shape::float_type, {2, 4}}, w));
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {5, 2}});
    p.add_instruction(migraphx::op::dot{}, l0, l1);

    auto prog = optimize_onnx("initializer_not_an_input.onnx");

    EXPECT(p == prog);
}

TEST_CASE(instance_norm_test)
{
    std::vector<size_t> dims{1, 2, 3, 3};
    migraphx::shape s1{migraphx::shape::float_type, dims};
    migraphx::shape s2{migraphx::shape::float_type, {2}};

    migraphx::program p;
    auto x     = p.add_parameter("0", s1);
    auto scale = p.add_parameter("1", s2);
    auto bias  = p.add_parameter("2", s2);

    auto mean            = p.add_instruction(migraphx::op::reduce_mean{{2, 3}}, x);
    auto mean_bcast      = p.add_instruction(migraphx::op::multibroadcast{dims}, mean);
    auto l0              = p.add_instruction(migraphx::op::sqdiff{}, x, mean_bcast);
    auto variance        = p.add_instruction(migraphx::op::reduce_mean{{2, 3}}, l0);
    auto l1              = p.add_instruction(migraphx::op::sub{}, x, mean_bcast);
    auto epsilon_literal = p.add_literal(1e-5f);
    auto epsilon_bcast   = p.add_instruction(migraphx::op::multibroadcast{dims}, epsilon_literal);
    auto variance_bcast  = p.add_instruction(migraphx::op::multibroadcast{dims}, variance);
    auto l2              = p.add_instruction(migraphx::op::add{}, variance_bcast, epsilon_bcast);
    auto l3              = p.add_instruction(migraphx::op::rsqrt{}, l2);
    auto l4              = p.add_instruction(migraphx::op::mul{}, l1, l3);
    auto scale_bcast     = p.add_instruction(migraphx::op::broadcast{1, dims}, scale);
    auto bias_bcast      = p.add_instruction(migraphx::op::broadcast{1, dims}, bias);
    auto l5              = p.add_instruction(migraphx::op::mul{}, l4, scale_bcast);
    p.add_instruction(migraphx::op::add{}, l5, bias_bcast);

    auto prog = optimize_onnx("instance_norm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(leaky_relu_test)
{
    migraphx::program p;
    float alpha = 0.01f;
    auto l0     = p.add_parameter("0", {migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::leaky_relu{alpha}, l0);

    auto prog = optimize_onnx("leaky_relu_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(log_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::log{}, input);

    auto prog = optimize_onnx("log_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(logsoftmax_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    int axis = 1;
    p.add_instruction(migraphx::op::logsoftmax{axis}, l0);
    auto prog = optimize_onnx("logsoftmax_test.onnx");

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
    auto prog = optimize_onnx("lrn_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_bmbm_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1  = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {5, 2, 1, 7, 8}});
    auto bl0 = p.add_instruction(migraphx::op::multibroadcast{{5, 2, 3, 6, 7}}, l0);
    auto bl1 = p.add_instruction(migraphx::op::multibroadcast{{5, 2, 3, 7, 8}}, l1);
    p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, bl0, bl1);

    auto prog = optimize_onnx("matmul_bmbm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_bmv_test)
{
    migraphx::program p;
    auto l0   = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1   = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl1  = p.add_instruction(migraphx::op::unsqueeze{{1}}, l1);
    auto bsl1 = p.add_instruction(migraphx::op::multibroadcast{{3, 7, 1}}, sl1);
    auto res  = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, l0, bsl1);
    p.add_instruction(migraphx::op::squeeze{{2}}, res);

    auto prog = optimize_onnx("matmul_bmv_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_mv_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {6, 7}});
    auto l1  = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl1 = p.add_instruction(migraphx::op::unsqueeze{{1}}, l1);
    auto res = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, l0, sl1);
    p.add_instruction(migraphx::op::squeeze{{1}}, res);

    auto prog = optimize_onnx("matmul_mv_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_vbm_test)
{
    migraphx::program p;
    auto l0   = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1   = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {5, 7, 8}});
    auto sl0  = p.add_instruction(migraphx::op::unsqueeze{{0}}, l0);
    auto bsl0 = p.add_instruction(migraphx::op::multibroadcast{{5, 1, 7}}, sl0);
    auto res  = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, bsl0, l1);
    p.add_instruction(migraphx::op::squeeze{{1}}, res);

    auto prog = optimize_onnx("matmul_vbm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_vm_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7, 8}});
    auto sl0 = p.add_instruction(migraphx::op::unsqueeze{{0}}, l0);
    auto res = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, sl0, l1);
    p.add_instruction(migraphx::op::squeeze{{0}}, res);

    auto prog = optimize_onnx("matmul_vm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_vv_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl0 = p.add_instruction(migraphx::op::unsqueeze{{0}}, l0);
    auto sl1 = p.add_instruction(migraphx::op::unsqueeze{{1}}, l1);
    auto res = p.add_instruction(migraphx::op::dot{1.0f, 0.0f}, sl0, sl1);
    auto sr0 = p.add_instruction(migraphx::op::squeeze{{0}}, res);
    p.add_instruction(migraphx::op::squeeze{{0}}, sr0);

    auto prog = optimize_onnx("matmul_vv_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmulinteger_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("1", migraphx::shape{migraphx::shape::int8_type, {3, 6, 16}});
    auto l1 = p.add_parameter("2", migraphx::shape{migraphx::shape::int8_type, {3, 16, 8}});
    p.add_instruction(migraphx::op::quant_dot{1, 0}, l0, l1);

    auto prog = optimize_onnx("matmulinteger_test.onnx");

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

    optimize_onnx("max_test.onnx");
}

TEST_CASE(maxpool_notset_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0, 1, 1};
    float val                 = std::numeric_limits<float>::lowest();
    auto ins_pad              = p.add_instruction(migraphx::op::pad{pads, val}, input);
    p.add_instruction(migraphx::op::pooling{"max", {0, 0}, {2, 2}, {6, 6}}, ins_pad);

    auto prog = optimize_onnx("maxpool_notset_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(maxpool_same_upper_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0, 1, 1};
    float val                 = std::numeric_limits<float>::lowest();
    auto ins_pad              = p.add_instruction(migraphx::op::pad{pads, val}, input);
    p.add_instruction(
        migraphx::op::pooling{"max", {0, 0}, {1, 1}, {2, 2}, migraphx::op::padding_mode_t::same},
        ins_pad);

    auto prog = optimize_onnx("maxpool_same_upper_test.onnx");

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

    optimize_onnx("min_test.onnx");
}

TEST_CASE(no_pad_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    p.add_instruction(migraphx::op::identity{}, l0);
    auto prog = optimize_onnx("no_pad_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(neg_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    auto input = p.add_parameter("0", s);
    auto ret   = p.add_instruction(migraphx::op::neg{}, input);
    p.add_return({ret});

    auto prog = migraphx::parse_onnx("neg_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(onehot_test)
{
    migraphx::program p;
    migraphx::shape s_ind{migraphx::shape::int32_type, {5, 2}};
    migraphx::shape s_val{migraphx::shape::half_type, {2}};
    p.add_literal(3);
    auto l_ind = p.add_parameter("indices", s_ind);
    auto l_val = p.add_parameter("values", s_val);
    migraphx::shape s_dep{migraphx::shape::half_type, {3, 3}};
    std::vector<float> data_dep{1, 0, 0, 0, 1, 0, 0, 0, 1};
    auto l_dep      = p.add_literal(migraphx::literal(s_dep, data_dep));
    auto gather_out = p.add_instruction(migraphx::op::gather{0}, l_dep, l_ind);
    auto tr_out     = p.add_instruction(migraphx::op::transpose{{2, 0, 1}}, gather_out);
    auto off_val    = p.add_instruction(migraphx::op::slice{{0}, {0}, {1}}, l_val);
    auto on_val     = p.add_instruction(migraphx::op::slice{{0}, {1}, {2}}, l_val);
    auto diff       = p.add_instruction(migraphx::op::sub{}, on_val, off_val);
    auto mb_off_val = p.add_instruction(migraphx::op::multibroadcast{{3, 5, 2}}, off_val);
    auto mb_diff    = p.add_instruction(migraphx::op::multibroadcast{{3, 5, 2}}, diff);
    auto mul        = p.add_instruction(migraphx::op::mul{}, tr_out, mb_diff);
    auto r          = p.add_instruction(migraphx::op::add{}, mul, mb_off_val);
    p.add_return({r});

    auto prog = migraphx::parse_onnx("onehot_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pad_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    p.add_instruction(migraphx::op::pad{{1, 1, 1, 1}}, l0);
    auto prog = optimize_onnx("pad_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pad_3arg_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    p.add_literal({migraphx::shape{migraphx::shape::float_type}, {1.0f}});
    p.add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 1, 2, 2}});
    auto r = p.add_instruction(migraphx::op::pad{{1, 1, 2, 2}, 1.0f}, l0);
    p.add_return({r});

    auto prog = migraphx::parse_onnx("pad_3arg_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pad_reflect_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    p.add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {0, 2, 0, 1}});
    auto l1 = p.add_instruction(migraphx::op::slice{{0, 1}, {0, 1}, {2, 2}}, l0);
    auto l2 = p.add_instruction(migraphx::op::slice{{0, 1}, {0, 0}, {2, 1}}, l0);
    auto l3 = p.add_instruction(migraphx::op::slice{{0, 1}, {0, 0}, {2, 1}}, l0);
    auto r  = p.add_instruction(migraphx::op::concat{1}, l2, l1, l0, l3);
    p.add_return({r});

    auto prog = migraphx::parse_onnx("pad_reflect_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pad_reflect_multiaxis_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    p.add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {0, 2, 2, 0}});
    auto l1 = p.add_instruction(migraphx::op::slice{{0, 1}, {0, 1}, {2, 2}}, l0);
    auto l2 = p.add_instruction(migraphx::op::slice{{0, 1}, {0, 2}, {2, 3}}, l0);
    auto l3 = p.add_instruction(migraphx::op::concat{1}, l2, l1, l0);
    auto l4 = p.add_instruction(migraphx::op::slice{{0, 1}, {0, 0}, {1, 5}}, l3);
    auto l5 = p.add_instruction(migraphx::op::slice{{0, 1}, {1, 0}, {2, 5}}, l3);
    auto r  = p.add_instruction(migraphx::op::concat{0}, l3, l4, l5);
    p.add_return({r});

    auto prog = migraphx::parse_onnx("pad_reflect_multiaxis_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pow_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    p.add_instruction(migraphx::op::pow{}, l0, l1);

    auto prog = optimize_onnx("pow_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(prelu_brcst_test)
{
    migraphx::program p;
    auto l0  = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 5}});
    auto bl1 = p.add_instruction(migraphx::op::multibroadcast{l0->get_shape().lens()}, l1);
    auto ret = p.add_instruction(migraphx::op::prelu{}, l0, bl1);
    p.add_return({ret});

    auto prog = migraphx::parse_onnx("prelu_brcst_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(range_test)
{
    migraphx::program p;
    p.add_literal(int64_t{10});
    p.add_literal(int64_t{6});
    p.add_literal(int64_t{-3});
    p.add_literal(migraphx::literal{{migraphx::shape::int64_type, {2}}, {10, 7}});

    auto prog = optimize_onnx("range_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(range_float_test)
{
    migraphx::program p;
    p.add_literal(float{2});
    p.add_literal(float{11});
    p.add_literal(float{2});
    p.add_literal(migraphx::literal{{migraphx::shape::float_type, {5}}, {2, 4, 6, 8, 10}});

    auto prog = optimize_onnx("range_float_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(recip_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3}});
    p.add_instruction(migraphx::op::recip{}, input);

    auto prog = optimize_onnx("recip_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducel1_test)
{
    migraphx::program p;
    auto l0     = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto abs_l0 = p.add_instruction(migraphx::op::abs{}, l0);
    auto sum_l0 = p.add_instruction(migraphx::op::reduce_sum{{-2}}, abs_l0);
    p.add_instruction(migraphx::op::squeeze{{-2}}, sum_l0);
    auto prog = optimize_onnx("reducel1_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducel2_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto square_l0 = p.add_instruction(migraphx::op::mul{}, l0, l0);
    auto sum_l0    = p.add_instruction(migraphx::op::reduce_sum{{-1}}, square_l0);
    auto squ_l0    = p.add_instruction(migraphx::op::squeeze{{-1}}, sum_l0);
    p.add_instruction(migraphx::op::sqrt{}, squ_l0);
    auto prog = optimize_onnx("reducel2_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reduce_log_sum_test)
{
    migraphx::program p;
    auto l0     = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto sum_l0 = p.add_instruction(migraphx::op::reduce_sum{{-3}}, l0);
    p.add_instruction(migraphx::op::log{}, sum_l0);
    auto prog = optimize_onnx("reduce_log_sum_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reduce_log_sum_exp_test)
{
    migraphx::program p;
    auto l0     = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto exp_l0 = p.add_instruction(migraphx::op::exp{}, l0);
    auto sum_l0 = p.add_instruction(migraphx::op::reduce_sum{{-4}}, exp_l0);
    p.add_instruction(migraphx::op::log{}, sum_l0);
    auto prog = optimize_onnx("reduce_log_sum_exp_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducemax_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    p.add_instruction(migraphx::op::reduce_max{{2}}, l0);
    auto prog = optimize_onnx("reducemax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducemean_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 = p.add_instruction(migraphx::op::reduce_mean{{2, 3}}, l0);
    p.add_instruction(migraphx::op::squeeze{{2, 3}}, l1);
    auto prog = optimize_onnx("reducemean_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducemean_keepdims_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    p.add_instruction(migraphx::op::reduce_mean{{2}}, l0);
    auto prog = optimize_onnx("reducemean_keepdims_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducemin_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 = p.add_instruction(migraphx::op::reduce_min{{2, 3}}, l0);
    p.add_instruction(migraphx::op::squeeze{{2, 3}}, l1);
    auto prog = optimize_onnx("reducemin_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reduceprod_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    p.add_instruction(migraphx::op::reduce_prod{{2}}, l0);
    auto prog = optimize_onnx("reduceprod_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 = p.add_instruction(migraphx::op::reduce_sum{{2}}, l0);
    p.add_instruction(migraphx::op::squeeze{{2}}, l1);
    auto prog = optimize_onnx("reducesum_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_multiaxis_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 = p.add_instruction(migraphx::op::reduce_sum{{2, 3}}, l0);
    p.add_instruction(migraphx::op::squeeze{{2, 3}}, l1);
    auto prog = optimize_onnx("reducesum_multiaxis_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_keepdims_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    p.add_instruction(migraphx::op::reduce_sum{{2, 3}}, l0);
    auto prog = optimize_onnx("reducesum_keepdims_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_square_test)
{
    migraphx::program p;
    auto l0     = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto squ_l0 = p.add_instruction(migraphx::op::mul{}, l0, l0);
    auto sum_l0 = p.add_instruction(migraphx::op::reduce_sum{{-2}}, squ_l0);
    p.add_instruction(migraphx::op::squeeze{{-2}}, sum_l0);
    auto prog = optimize_onnx("reducesum_square_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reshape_test)
{
    migraphx::program p;
    migraphx::op::reshape op;
    std::vector<int64_t> reshape_dims{3, 8};
    p.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, reshape_dims});
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});
    op.dims = reshape_dims;
    p.add_instruction(op, l0);
    p.add_instruction(op, l0);
    auto prog = optimize_onnx("reshape_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reshape_non_standard_test)
{
    migraphx::program p;
    migraphx::op::reshape op;
    std::vector<int64_t> reshape_dims{4, 3, 2};
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
    auto x      = p.add_parameter("x", s);
    auto tran_x = p.add_instruction(migraphx::op::transpose{{0, 2, 1}}, x);
    auto cont_x = p.add_instruction(migraphx::op::contiguous{}, tran_x);
    p.add_instruction(migraphx::op::reshape{{4, 3, 2}}, cont_x);
    auto prog = optimize_onnx("reshape_non_standard_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(round_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::double_type, {10, 5}});
    p.add_instruction(migraphx::op::round{}, input);

    auto prog = optimize_onnx("round_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(shape_test)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3, 4, 5, 6}};
    auto l0 = p.add_parameter("x", s);
    migraphx::shape s_shape{migraphx::shape::int64_type, {4}};
    p.add_literal(s_shape, l0->get_shape().lens());
    auto prog = optimize_onnx("shape_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(shape_gather_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {7, 3, 10}});
    migraphx::shape const_shape{migraphx::shape::int32_type, {1}};
    auto l2 = p.add_literal(migraphx::literal{const_shape, {1}});
    auto l1 =
        p.add_literal(migraphx::shape{migraphx::shape::int64_type, {3}}, l0->get_shape().lens());
    int axis = 0;
    p.add_instruction(migraphx::op::gather{axis}, l1, l2);
    auto prog = optimize_onnx("shape_gather_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sign_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::double_type, {10, 5}});
    p.add_instruction(migraphx::op::sign{}, input);

    auto prog = optimize_onnx("sign_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sin_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::sin{}, input);

    auto prog = optimize_onnx("sin_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sinh_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::sinh{}, input);

    auto prog = optimize_onnx("sinh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(slice_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 2}});
    p.add_instruction(migraphx::op::slice{{0, 1}, {1, 0}, {2, 2}}, l0);
    auto prog = optimize_onnx("slice_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(slice_3arg_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 5}});
    p.add_literal({{migraphx::shape::int32_type, {2}}, {0, 0}});
    p.add_literal({{migraphx::shape::int32_type, {2}}, {2, 5}});
    auto ret = p.add_instruction(migraphx::op::slice{{0, 1}, {0, 0}, {2, 5}}, l0);
    p.add_return({ret});

    auto prog = migraphx::parse_onnx("slice_3arg_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(slice_5arg_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 5}});
    p.add_literal({{migraphx::shape::int32_type, {2}}, {1, 1}});
    p.add_literal({{migraphx::shape::int32_type, {2}}, {-1, -2}});
    p.add_literal({{migraphx::shape::int32_type, {2}}, {-1, -1}});
    p.add_literal({{migraphx::shape::int32_type, {2}}, {-5, -3}});
    auto ret = p.add_instruction(migraphx::op::slice{{-1, -2}, {-5, -3}, {-1, -1}}, l0);
    p.add_return({ret});

    auto prog = migraphx::parse_onnx("slice_5arg_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(slice_max_end_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {10, 20}});
    p.add_instruction(migraphx::op::slice{{0, 1}, {1, 2}, {3000000000, -1}}, l0);
    auto prog = optimize_onnx("slice_max_end_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(softmax_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    p.add_instruction(migraphx::op::softmax{1}, l0);
    auto prog = optimize_onnx("softmax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(split_minus_axis_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    auto r1    = p.add_instruction(migraphx::op::slice{{-1}, {0}, {5}}, input);
    auto r2    = p.add_instruction(migraphx::op::slice{{-1}, {5}, {10}}, input);
    auto r3    = p.add_instruction(migraphx::op::slice{{-1}, {10}, {15}}, input);
    p.add_return({r1, r2, r3});

    auto prog = migraphx::parse_onnx("split_minus_axis_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(split_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    auto r1    = p.add_instruction(migraphx::op::slice{{1}, {0}, {7}}, input);
    auto r2    = p.add_instruction(migraphx::op::slice{{1}, {7}, {11}}, input);
    auto r3    = p.add_instruction(migraphx::op::slice{{1}, {11}, {15}}, input);
    p.add_return({r1, r2, r3});

    auto prog = migraphx::parse_onnx("split_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(split_test_default)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    auto r1    = p.add_instruction(migraphx::op::slice{{0}, {0}, {5}}, input);
    auto r2    = p.add_instruction(migraphx::op::slice{{0}, {5}, {10}}, input);
    p.add_return({r1, r2});

    auto prog = migraphx::parse_onnx("split_test_default.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sqrt_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    p.add_instruction(migraphx::op::sqrt{}, input);

    auto prog = optimize_onnx("sqrt_test.onnx");
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
    auto prog = optimize_onnx("squeeze_unsqueeze_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sub_bcast_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2 = p.add_instruction(migraphx::op::broadcast{1, l0->get_shape().lens()}, l1);
    p.add_instruction(migraphx::op::sub{}, l0, l2);

    auto prog = optimize_onnx("sub_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sub_scalar_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1}});
    auto m1 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4, 5}}, l1);
    p.add_instruction(migraphx::op::sub{}, l0, m1);
    auto prog = optimize_onnx("sub_scalar_test.onnx");

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

    auto prog = optimize_onnx("sum_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(tan_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    p.add_instruction(migraphx::op::tan{}, input);

    auto prog = optimize_onnx("tan_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(tanh_test)
{
    migraphx::program p;
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    p.add_instruction(migraphx::op::tanh{}, input);

    auto prog = optimize_onnx("tanh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(tile_test)
{
    migraphx::program p;
    p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, {1, 2}});
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    p.add_instruction(migraphx::op::concat{1}, input, input);

    auto prog = optimize_onnx("tile_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(tile_test_3x2)
{
    migraphx::program p;
    p.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, {3, 2}});
    auto input = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    auto l0    = p.add_instruction(migraphx::op::concat{0}, input, input);
    auto l1    = p.add_instruction(migraphx::op::concat{0}, l0, input);
    p.add_instruction(migraphx::op::concat{1}, l1, l1);

    auto prog = optimize_onnx("tile_test_3x2.onnx");

    EXPECT(p == prog);
}

TEST_CASE(transpose_test)
{
    migraphx::program p;
    auto input = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    std::vector<int64_t> perm{0, 3, 1, 2};
    p.add_instruction(migraphx::op::transpose{perm}, input);

    auto prog = optimize_onnx("transpose_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(transpose_gather_test)
{
    migraphx::program p;
    auto make_contiguous = [&p](migraphx::instruction_ref ins) {
        if(ins->get_shape().standard())
        {
            return ins;
        }

        return p.add_instruction(migraphx::op::contiguous{}, ins);
    };

    auto data = p.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 5, 4, 6}});
    auto ind =
        p.add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, {2, 4, 3, 5}});
    auto tr_data = p.add_instruction(migraphx::op::transpose{{0, 2, 1, 3}}, data);
    auto tr_ind  = p.add_instruction(migraphx::op::transpose{{0, 2, 1, 3}}, ind);
    int axis     = 1;
    p.add_instruction(
        migraphx::op::gather{axis}, make_contiguous(tr_data), make_contiguous(tr_ind));

    auto prog = optimize_onnx("transpose_gather_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(undefined_test)
{
    migraphx::program p;
    p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_instruction(migraphx::op::undefined{});
    auto l2 = p.add_instruction(migraphx::op::identity{}, l1);
    p.add_return({l2});

    auto prog = migraphx::parse_onnx("undefined_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(unknown_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2 = p.add_instruction(migraphx::op::unknown{"Unknown"}, l0, l1);
    p.add_instruction(migraphx::op::unknown{"Unknown"}, l2);
    auto prog = optimize_onnx("unknown_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(unknown_aten_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("unknown_aten_test.onnx"); }));
}

TEST_CASE(unknown_test_throw)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("unknown_test.onnx"); }));
}

TEST_CASE(unknown_test_throw_print_error)
{
    migraphx::onnx_options options;
    options.print_program_on_error = true;
    EXPECT(test::throws([&] { migraphx::parse_onnx("unknown_test.onnx", options); }));
}

TEST_CASE(variable_batch_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::identity{}, l0);
    auto prog = optimize_onnx("variable_batch_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 16, 16}});
    auto r  = p.add_instruction(migraphx::op::identity{}, l0);
    p.add_return({r});

    migraphx::onnx_options options;
    options.default_dim_value = 2;

    auto prog = migraphx::parse_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_leq_zero_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::add{}, l0, l1);
    auto prog = optimize_onnx("variable_batch_leq_zero_test.onnx");

    EXPECT(p == prog);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
