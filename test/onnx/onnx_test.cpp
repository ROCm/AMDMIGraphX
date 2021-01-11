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
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include "test.hpp"

migraphx::program optimize_onnx(const std::string& name, bool eliminate_deadcode = false)
{
    migraphx::onnx_options options;
    options.skip_unknown_operators = true;
    auto prog                      = migraphx::parse_onnx(name, options);
    auto* mm                       = prog.get_main_module();
    if(eliminate_deadcode)
        migraphx::run_passes(*mm, {migraphx::dead_code_elimination{}});

    // remove the last identity instruction
    auto last_ins = std::prev(mm->end());
    if(last_ins->name() == "@return")
    {
        mm->remove_instruction(last_ins);
    }

    return prog;
}

TEST_CASE(acos_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("acos"), input);

    auto prog = optimize_onnx("acos_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(acosh_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("acosh"), input);

    auto prog = optimize_onnx("acosh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(add_bcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"dims", l0->get_shape().lens()}}), l1);
    mm->add_instruction(migraphx::make_op("add"), l0, l2);

    auto prog = optimize_onnx("add_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(add_fp16_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {1.5}});
    auto l1 =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {2.5}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    auto prog = optimize_onnx("add_fp16_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(add_scalar_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::uint8_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::uint8_type});
    auto m1  = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {2, 3, 4, 5}}}), l1);
    auto r = mm->add_instruction(migraphx::make_op("add"), l0, m1);
    mm->add_return({r});
    auto prog = migraphx::parse_onnx("add_scalar_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(argmax_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto ins = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), ins);
    auto prog = optimize_onnx("argmax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(argmin_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto ins = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 3}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), ins);
    auto prog = optimize_onnx("argmin_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(asin_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("asin"), input);

    auto prog = optimize_onnx("asin_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(asinh_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("asinh"), input);

    auto prog = optimize_onnx("asinh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(atan_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("atan"), input);

    auto prog = optimize_onnx("atan_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(atanh_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("atanh"), input);

    auto prog = optimize_onnx("atanh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(averagepool_1d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 5}});
    mm->add_instruction(
        migraphx::make_op(
            "pooling", {{"mode", "average"}, {"padding", {0}}, {"stride", {1}}, {"lengths", {3}}}),
        l0);

    auto prog = optimize_onnx("averagepool_1d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(averagepool_3d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 5, 5, 5}});
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", "average"},
                                           {"padding", {0, 0, 0}},
                                           {"stride", {1, 1, 1}},
                                           {"lengths", {3, 3, 3}}}),
                        l0);

    auto prog = optimize_onnx("averagepool_3d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(averagepool_notset_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    auto ins   = mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "average"}, {"padding", {2, 2}}, {"stride", {2, 2}}, {"lengths", {6, 6}}}),
        input);
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {1, 1}}, {"ends", {2, 2}}}), ins);
    mm->add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_notset_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(averagepool_nt_cip_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0, 1, 1};
    auto ins_pad = mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), input);
    auto ret     = mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "average"}, {"padding", {0, 0}}, {"stride", {2, 2}}, {"lengths", {6, 6}}}),
        ins_pad);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("averagepool_nt_cip_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(averagepool_same_lower_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    auto ins   = mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "average"}, {"padding", {1, 1}}, {"stride", {1, 1}}, {"lengths", {2, 2}}}),
        input);
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {0, 0}}, {"ends", {5, 5}}}), ins);
    mm->add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_same_lower_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(averagepool_sl_cip_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    std::vector<int64_t> pads = {0, 0, 1, 1, 0, 0, 0, 0};
    auto ins_pad = mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), input);
    auto ret     = mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "average"}, {"padding", {0, 0}}, {"stride", {1, 1}}, {"lengths", {2, 2}}}),
        ins_pad);
    mm->add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_sl_cip_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(averagepool_same_upper_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    auto ins   = mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "average"}, {"padding", {1, 1}}, {"stride", {1, 1}}, {"lengths", {2, 2}}}),
        input);
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {1, 1}}, {"ends", {6, 6}}}), ins);
    mm->add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_same_upper_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(batchnorm_1d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 5}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {3}});
    auto l2  = mm->add_parameter("2", {migraphx::shape::float_type, {3}});
    auto l3  = mm->add_parameter("3", {migraphx::shape::float_type, {3}});
    auto l4  = mm->add_parameter("4", {migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("batch_norm_inference"), l0, l1, l2, l3, l4);

    auto prog = optimize_onnx("batchnorm_1d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(batchnorm_3d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 5, 5, 5}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {3}});
    auto l2  = mm->add_parameter("2", {migraphx::shape::float_type, {3}});
    auto l3  = mm->add_parameter("3", {migraphx::shape::float_type, {3}});
    auto l4  = mm->add_parameter("4", {migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("batch_norm_inference"), l0, l1, l2, l3, l4);

    auto prog = optimize_onnx("batchnorm_3d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(cast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_parameter("x", migraphx::shape{migraphx::shape::half_type, {10}});
    mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l);

    auto prog = optimize_onnx("cast_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(ceil_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("ceil"), input);

    auto prog = optimize_onnx("ceil_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto l0      = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {3}}}), min_val);
    max_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {3}}}), max_val);
    mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    auto prog = optimize_onnx("clip_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test_op11_max_only)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto max_val = mm->add_literal(0.0f);
    auto l0      = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("undefined"));
    max_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {3}}}), max_val);
    auto r = mm->add_instruction(migraphx::make_op("min"), l0, max_val);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("clip_test_op11_max_only.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test_op11)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    auto l0      = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    min_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {3}}}), min_val);
    max_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {3}}}), max_val);
    mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    auto prog = optimize_onnx("clip_test_op11.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test_op11_min_only)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto min_val = mm->add_literal(0.0f);
    auto l0      = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    min_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {3}}}), min_val);
    mm->add_instruction(migraphx::make_op("max"), l0, min_val);
    auto prog = optimize_onnx("clip_test_op11_min_only.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test_op11_no_args)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_onnx("clip_test_op11_no_args.onnx");

    EXPECT(p == prog);
}

TEST_CASE(clip_test_op11_no_args1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("undefined"));
    auto r = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});
    auto prog = migraphx::parse_onnx("clip_test_op11_no_args1.onnx");

    EXPECT(p == prog);
}

TEST_CASE(concat_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4, 3}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7, 4, 3}});
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), l0, l1);
    auto prog = optimize_onnx("concat_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0, 1, 2}});
    auto prog = optimize_onnx("constant_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_fill_test)
{

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> value(s.elements(), 1.0);
    mm->add_literal(migraphx::literal{s, value});
    auto prog = optimize_onnx("constant_fill_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_fill_input_as_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {2}}, {2, 3}});
    std::vector<std::size_t> dims(l0->get_shape().elements());
    migraphx::literal ls = l0->get_literal();
    ls.visit([&](auto s) { dims.assign(s.begin(), s.end()); });
    migraphx::shape s{migraphx::shape::float_type, dims};
    std::vector<float> value(s.elements(), 1.0);
    mm->add_literal(migraphx::literal{s, value});
    auto prog = optimize_onnx("constant_fill_input_as_shape_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_scalar_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1}}, {1}});
    auto prog = optimize_onnx("constant_scalar_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(const_of_shape_empty_input_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal());
    migraphx::shape s(migraphx::shape::int64_type, {1}, {0});
    std::vector<int64_t> vec(s.elements(), 10);
    mm->add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_empty_input_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(const_of_shape_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss(migraphx::shape::int32_type, {3});
    mm->add_literal(migraphx::literal(ss, {2, 3, 4}));
    migraphx::shape s(migraphx::shape::float_type, {2, 3, 4});
    std::vector<float> vec(s.elements(), 10.0f);
    mm->add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_float_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(const_of_shape_int64_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss(migraphx::shape::int32_type, {3});
    mm->add_literal(migraphx::literal(ss, {2, 3, 4}));
    migraphx::shape s(migraphx::shape::int64_type, {2, 3, 4});
    std::vector<int64_t> vec(s.elements(), 10);
    mm->add_literal(migraphx::literal(s, vec));

    auto prog = optimize_onnx("const_of_shape_int64_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(const_of_shape_no_value_attr_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss(migraphx::shape::int32_type, {3});
    mm->add_literal(migraphx::literal(ss, {2, 3, 4}));
    migraphx::shape s(migraphx::shape::float_type, {2, 3, 4});
    std::vector<float> vec(s.elements(), 0.0f);
    mm->add_literal(migraphx::literal(s, vec));

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
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 5}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 3}});
    mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        l0,
        l1);

    auto prog = optimize_onnx("conv_1d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_3d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 5, 5, 5}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 3, 3, 3}});
    mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        l0,
        l1);

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
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 3, 3}});
    migraphx::op::convolution op;
    op.padding      = {1, 1};
    op.padding_mode = migraphx::op::padding_mode_t::same;
    mm->add_instruction(op, l0, l1);

    auto prog = optimize_onnx("conv_autopad_same_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_bias_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = mm->add_instruction(migraphx::make_op("convolution"), l0, l1);
    auto l4       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"dims", l3->get_shape().lens()}}), l2);
    mm->add_instruction(migraphx::make_op("add"), l3, l4);

    auto prog = optimize_onnx("conv_bias_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_bn_relu_maxpool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1  = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2  = mm->add_parameter("2", {migraphx::shape::float_type, {1}});

    auto p3       = mm->add_parameter("3", {migraphx::shape::float_type, {1}});
    auto p4       = mm->add_parameter("4", {migraphx::shape::float_type, {1}});
    auto p5       = mm->add_parameter("5", {migraphx::shape::float_type, {1}});
    auto p6       = mm->add_parameter("6", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = mm->add_instruction(migraphx::make_op("convolution"), l0, l1);
    auto l4       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"dims", l3->get_shape().lens()}}), l2);
    auto l5 = mm->add_instruction(migraphx::make_op("add"), l3, l4);
    auto l6 = mm->add_instruction(
        migraphx::make_op("batch_norm_inference", {{"epsilon", 1.0e-5f}}), l5, p3, p4, p5, p6);
    auto l7 = mm->add_instruction(migraphx::make_op("relu"), l6);
    mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "max"}, {"padding", {0, 0}}, {"stride", {2, 2}}, {"lengths", {2, 2}}}),
        l7);

    auto prog = optimize_onnx("conv_bn_relu_maxpool_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_relu_maxpool_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = mm->add_instruction(migraphx::make_op("convolution"), l0, l1);
    auto l4       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"dims", l3->get_shape().lens()}}), l2);
    auto l5 = mm->add_instruction(migraphx::make_op("add"), l3, l4);
    auto l6 = mm->add_instruction(migraphx::make_op("relu"), l5);
    mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "max"}, {"padding", {0, 0}}, {"stride", {2, 2}}, {"lengths", {2, 2}}}),
        l6);

    auto prog = optimize_onnx("conv_relu_maxpool_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(conv_relu_maxpool_x2_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = mm->add_parameter("1", {migraphx::shape::float_type, {5, 3, 5, 5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::float_type, {5}});
    uint64_t axis = 1;
    auto l3       = mm->add_instruction(migraphx::make_op("convolution"), l0, l1);
    auto l4       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"dims", l3->get_shape().lens()}}), l2);
    auto l5 = mm->add_instruction(migraphx::make_op("add"), l3, l4);
    auto l6 = mm->add_instruction(migraphx::make_op("relu"), l5);
    auto l7 = mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "max"}, {"padding", {0, 0}}, {"stride", {2, 2}}, {"lengths", {2, 2}}}),
        l6);

    auto l8  = mm->add_parameter("3", {migraphx::shape::float_type, {1, 5, 5, 5}});
    auto l9  = mm->add_parameter("4", {migraphx::shape::float_type, {1}});
    auto l10 = mm->add_instruction(migraphx::make_op("convolution"), l7, l8);
    auto l11 = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"dims", l10->get_shape().lens()}}), l9);
    auto l12 = mm->add_instruction(migraphx::make_op("add"), l10, l11);
    auto l13 = mm->add_instruction(migraphx::make_op("relu"), l12);
    mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "max"}, {"padding", {0, 0}}, {"stride", {2, 2}}, {"lengths", {2, 2}}}),
        l13);

    auto prog = optimize_onnx("conv_relu_maxpool_x2_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(convinteger_bias_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("0", {migraphx::shape::int8_type, {1, 3, 32, 32}});
    auto l1       = mm->add_parameter("1", {migraphx::shape::int8_type, {1, 3, 5, 5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::int32_type, {1}});
    uint64_t axis = 1;
    auto l3       = mm->add_instruction(migraphx::make_op("quant_convolution"), l0, l1);
    auto l4       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"dims", l3->get_shape().lens()}}), l2);
    mm->add_instruction(migraphx::make_op("add"), l3, l4);

    auto prog = optimize_onnx("convinteger_bias_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(cos_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("cos"), input);

    auto prog = optimize_onnx("cos_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(cosh_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    mm->add_instruction(migraphx::make_op("cosh"), input);

    auto prog = optimize_onnx("cosh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(deconv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});
    mm->add_instruction(migraphx::make_op("deconvolution"), l0, l1);

    auto prog = optimize_onnx("deconv_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_bias_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1       = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l2       = mm->add_parameter("b", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = mm->add_instruction(migraphx::make_op("deconvolution"), l0, l1);
    auto l4       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"dims", l3->get_shape().lens()}}), l2);
    mm->add_instruction(migraphx::make_op("add"), l3, l4);

    auto prog = optimize_onnx("deconv_bias_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_input_pads_strides_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    mm->add_instruction(
        migraphx::make_op("deconvolution", {{"padding", {1, 1}}, {"stride", {3, 2}}}), l0, l1);

    auto prog = optimize_onnx("deconv_input_pads_strides_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_input_pads_asymm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("deconvolution", {{"padding", {0, 0}}, {"stride", {3, 2}}}), l0, l1);
    mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {0, 0}}, {"ends", {8, 6}}}), l2);

    auto prog = optimize_onnx("deconv_input_pads_asymm_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_input_pads_asymm_1d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("deconvolution", {{"padding", {0}}, {"stride", {2}}, {"dilation", {1}}}),
        l0,
        l1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {6}}}),
                        l2);

    auto prog = optimize_onnx("deconv_input_pads_asymm_1d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_output_padding_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("deconvolution", {{"padding", {0, 0}}, {"stride", {3, 2}}}), l0, l1);
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 1, 1}}}), l2);

    auto prog = optimize_onnx("deconv_output_padding_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_output_padding_3d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3, 3}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("deconvolution",
                          {{"padding", {0, 0, 0}}, {"stride", {3, 2, 2}}, {"dilation", {1, 1, 1}}}),
        l0,
        l1);
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 0, 1, 1, 1}}}), l2);

    auto prog = optimize_onnx("deconv_output_padding_3d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_output_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("deconvolution", {{"padding", {0, 0}}, {"stride", {3, 2}}}), l0, l1);
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 1, 1}}}), l2);

    auto prog = optimize_onnx("deconv_output_shape_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(deconv_output_shape_3d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3, 3}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 3, 3, 3}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("deconvolution",
                          {{"padding", {0, 0, 0}}, {"stride", {3, 2, 2}}, {"dilation", {1, 1, 1}}}),
        l0,
        l1);
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 0, 1, 1, 1}}}), l2);

    auto prog = optimize_onnx("deconv_output_shape_3d_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(dropout_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}});
    auto out   = mm->add_instruction(migraphx::make_op("identity"), input);
    migraphx::shape s{migraphx::shape::bool_type, {1, 3, 2, 2}};
    std::vector<int8_t> vec(s.elements(), 1);
    mm->add_literal(migraphx::literal(s, vec));
    mm->add_return({out});

    auto prog = migraphx::parse_onnx("dropout_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(elu_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("elu", {{"alpha", 0.01}}), input);

    auto prog = optimize_onnx("elu_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(embedding_bag_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("weight", migraphx::shape{migraphx::shape::float_type, {4, 2}});
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {3}}, {1, 0, 2}};
    auto l1 = mm->add_literal(l);
    mm->add_literal(0);
    auto l4 = mm->add_instruction(migraphx::make_op("gather"), l0, l1);
    auto r1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), l4);
    auto l5 = mm->add_instruction(migraphx::make_op("gather"), l0, l1);
    auto r2 = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0}}}), l5);
    auto l6 = mm->add_instruction(migraphx::make_op("gather"), l0, l1);
    auto r3 = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0}}}), l6);
    mm->add_return({r1, r2, r3});

    auto prog = migraphx::parse_onnx("embedding_bag_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(embedding_bag_offset_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("embedding_bag_offset_test.onnx"); }));
}

TEST_CASE(equal_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto input1 = mm->add_literal(migraphx::literal(s, data));
    auto input2 = mm->add_parameter("x2", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    auto eq     = mm->add_instruction(migraphx::make_op("equal"), input1, input2);
    auto ret    = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        eq);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("equal_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(equal_bool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sf{migraphx::shape::float_type, {2, 3}};
    migraphx::shape sb{migraphx::shape::bool_type, {2, 3}};

    auto input1 = mm->add_parameter("x1", sf);
    auto input2 = mm->add_parameter("x2", sb);
    auto cin1   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        input1);
    auto ret = mm->add_instruction(migraphx::make_op("equal"), cin1, input2);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("equal_bool_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(erf_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    mm->add_instruction(migraphx::make_op("erf"), input);

    auto prog = optimize_onnx("erf_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(exp_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("exp"), input);

    auto prog = optimize_onnx("exp_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(expand_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s(migraphx::shape::float_type, {3, 1, 1});
    auto param = mm->add_parameter("x", s);
    migraphx::shape ss(migraphx::shape::int32_type, {4});
    mm->add_literal(migraphx::literal(ss, {2, 3, 4, 5}));
    mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {2, 3, 4, 5}}}),
                        param);

    auto prog = optimize_onnx("expand_test.onnx");
    EXPECT(p == prog);
}

migraphx::program create_external_data_prog()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s(migraphx::shape::float_type, {1, 1, 224, 224});
    migraphx::shape s2(migraphx::shape::float_type, {10, 1, 11, 11});
    std::vector<float> weight_data(1210, 1);
    std::vector<float> bias_data(10, 1);
    auto bias = mm->add_literal(migraphx::literal({migraphx::shape::float_type, {10}}, bias_data));
    auto weights    = mm->add_literal(migraphx::literal(s2, weight_data));
    auto param      = mm->add_parameter("input", s);
    auto conv       = mm->add_instruction(migraphx::make_op("convolution"), param, weights);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 10, 214, 214}}}), bias);
    mm->add_instruction(migraphx::make_op("add"), conv, bias_bcast);
    return p;
}

TEST_CASE(external_data_test)
{
    migraphx::program p = create_external_data_prog();

    auto prog = optimize_onnx("external_data_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(external_data_diff_path_test)
{
    migraphx::program p = create_external_data_prog();

    auto prog = optimize_onnx("ext_path/external_data_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(flatten_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    mm->add_instruction(migraphx::make_op("flatten", {{"axis", 2}}), l0);
    mm->add_instruction(migraphx::make_op("flatten", {{"axis", 1}}), l0);
    auto prog = optimize_onnx("flatten_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(floor_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("floor"), input);

    auto prog = optimize_onnx("floor_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gather_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 = mm->add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, {2, 3}});
    int axis = 1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), l0, l1);
    auto prog = optimize_onnx("gather_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gather_elements_axis0_test)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("data", {migraphx::shape::float_type, {3, 4}});
    auto indices = mm->add_parameter("indices", {migraphx::shape::int32_type, {2, 3}});
    std::vector<int> ind_indices{0, 1, 2, 4, 5, 6};
    std::vector<int> ind_axis_indices{0, 0, 0, 1, 1, 1};
    migraphx::shape ind_s{migraphx::shape::int32_type, {2, 3}};
    auto l_data_indices =
        mm->add_literal(migraphx::literal{ind_s, ind_indices.begin(), ind_indices.end()});
    auto l_ind_axis_indices =
        mm->add_literal(migraphx::literal{ind_s, ind_axis_indices.begin(), ind_axis_indices.end()});
    auto l_stride = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {1}}, {4}});

    auto rsp_data    = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
    auto lbst_stride = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", ind_s.lens()}}), l_stride);
    auto axis_delta = mm->add_instruction(migraphx::make_op("sub"), indices, l_ind_axis_indices);
    auto mul_delta  = mm->add_instruction(migraphx::make_op("mul"), axis_delta, lbst_stride);
    auto ind        = mm->add_instruction(migraphx::make_op("add"), l_data_indices, mul_delta);
    auto ret = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp_data, ind);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("gather_elements_axis0_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gather_elements_axis1_test)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("data", {migraphx::shape::float_type, {3, 4}});
    auto indices = mm->add_parameter("indices", {migraphx::shape::int32_type, {2, 3}});
    std::vector<int> ind_indices{0, 1, 2, 4, 5, 6};
    std::vector<int> ind_axis_indices{0, 1, 2, 0, 1, 2};
    migraphx::shape ind_s{migraphx::shape::int32_type, {2, 3}};
    auto l_data_indices =
        mm->add_literal(migraphx::literal{ind_s, ind_indices.begin(), ind_indices.end()});
    auto l_ind_axis_indices =
        mm->add_literal(migraphx::literal{ind_s, ind_axis_indices.begin(), ind_axis_indices.end()});
    auto l_stride = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {1}}, {1}});

    auto rsp_data    = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
    auto lbst_stride = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", ind_s.lens()}}), l_stride);
    auto axis_delta = mm->add_instruction(migraphx::make_op("sub"), indices, l_ind_axis_indices);
    auto mul_delta  = mm->add_instruction(migraphx::make_op("mul"), axis_delta, lbst_stride);
    auto ind        = mm->add_instruction(migraphx::make_op("add"), l_data_indices, mul_delta);
    auto ret = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp_data, ind);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("gather_elements_axis1_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gemm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 7}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {11, 5}});
    auto l2  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type});
    auto t0  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l0);
    auto t1  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l1);
    auto bl2 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {7, 11}}}), l2);
    auto alpha = 2.f;
    auto beta  = 2.0f;
    mm->add_instruction(migraphx::make_op("dot", {{"alpha", alpha}, {"beta", beta}}), t0, t1, bl2);
    auto prog = optimize_onnx("gemm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gemm_ex_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto l0    = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 1, 8, 6}});
    auto l1    = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 1, 8, 7}});
    auto l2    = mm->add_parameter("3", migraphx::shape{migraphx::shape::float_type, {1, 1, 6, 7}});
    auto t0    = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 3, 2}}}), l0);
    auto alpha = 0.5f;
    auto beta  = 0.8f;
    mm->add_instruction(migraphx::make_op("dot", {{"alpha", alpha}, {"beta", beta}}), t0, l1, l2);
    auto prog = optimize_onnx("gemm_ex_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(gemm_ex_brcst_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 6}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 7}});
    auto l2  = mm->add_parameter("3", migraphx::shape{migraphx::shape::float_type, {1, 1, 6, 1}});
    auto t0  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 1, 3, 2}}}), l0);
    std::vector<std::size_t> out_lens{1, 1, 6, 7};
    auto t2 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", out_lens}}), l2);
    auto alpha = 0.5f;
    auto beta  = 0.8f;
    mm->add_instruction(migraphx::make_op("dot", {{"alpha", alpha}, {"beta", beta}}), t0, l1, t2);
    auto prog = optimize_onnx("gemm_ex_brcst_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(globalavgpool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto op    = migraphx::op::pooling{"average"};
    auto lens  = input->get_shape().lens();
    op.lengths = {lens[2], lens[3]};
    mm->add_instruction(op, input);

    auto prog = optimize_onnx("globalavgpool_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(globalmaxpool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto op    = migraphx::op::pooling{"max"};
    auto lens  = input->get_shape().lens();
    op.lengths = {lens[2], lens[3]};
    mm->add_instruction(op, input);

    auto prog = optimize_onnx("globalmaxpool_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(greater_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto input1 = mm->add_literal(migraphx::literal(s, data));
    auto input2 = mm->add_parameter("x2", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    auto gr     = mm->add_instruction(migraphx::make_op("greater"), input1, input2);
    auto ret    = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        gr);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("greater_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(greater_bool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sf{migraphx::shape::float_type, {2, 3}};
    migraphx::shape sb{migraphx::shape::bool_type, {2, 3}};

    auto input1 = mm->add_parameter("x1", sf);
    auto input2 = mm->add_parameter("x2", sb);
    auto cin1   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        input1);
    auto ret = mm->add_instruction(migraphx::make_op("greater"), cin1, input2);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("greater_bool_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(group_conv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 4, 16, 16}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 1, 3, 3}});
    migraphx::op::convolution op;
    op.group = 4;
    mm->add_instruction(op, l0, l1);
    auto prog = optimize_onnx("group_conv_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(imagescaler_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 16, 16}};
    auto l0        = mm->add_parameter("0", s);
    auto scale_val = mm->add_literal(0.5f);
    auto bias_vals = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0.01, 0.02, 0.03}});
    auto scaled_tensor = mm->add_instruction(
        migraphx::make_op("scalar", {{"scalar_bcst_dims", s.lens()}}), scale_val);
    auto img_scaled = mm->add_instruction(migraphx::make_op("mul"), l0, scaled_tensor);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"dims", s.lens()}}), bias_vals);
    mm->add_instruction(migraphx::make_op("add"), img_scaled, bias_bcast);

    auto prog = optimize_onnx("imagescaler_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(imagescaler_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {1, 3, 16, 16}};
    auto l0 = mm->add_parameter("0", s);
    auto scale_val =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {0.5f}});
    auto bias_vals = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::half_type, {3}}, {0.01, 0.02, 0.03}});
    auto scaled_tensor = mm->add_instruction(
        migraphx::make_op("scalar", {{"scalar_bcst_dims", s.lens()}}), scale_val);
    auto img_scaled = mm->add_instruction(migraphx::make_op("mul"), l0, scaled_tensor);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"dims", s.lens()}}), bias_vals);
    mm->add_instruction(migraphx::make_op("add"), img_scaled, bias_bcast);

    auto prog = optimize_onnx("imagescaler_half_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(implicit_add_bcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4, 1}});
    auto l3  = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {2, 3, 4, 5}}}), l1);
    mm->add_instruction(migraphx::make_op("add"), l0, l3);

    auto prog = optimize_onnx("implicit_add_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(implicit_add_bcast_user_input_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 5, 1}});
    auto l3  = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {3, 4, 5, 6}}}), l1);
    auto r = mm->add_instruction(migraphx::make_op("add"), l0, l3);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.map_input_dims["0"] = {3, 4, 5, 6};
    options.map_input_dims["1"] = {4, 5, 1};
    auto prog                   = migraphx::parse_onnx("implicit_add_bcast_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(implicit_pow_bcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4, 1}});
    auto l3  = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {2, 3, 4, 5}}}), l1);
    mm->add_instruction(migraphx::make_op("pow"), l0, l3);

    auto prog = optimize_onnx("implicit_pow_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(implicit_sub_bcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::uint64_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::uint64_type, {4, 5}});
    auto l3  = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {2, 3, 4, 5}}}), l1);
    mm->add_instruction(migraphx::make_op("sub"), l0, l3);

    auto prog = optimize_onnx("implicit_sub_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(initializer_not_an_input)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> w = {1, 2, 3, 4, 5, 6, 7, 8};
    auto l1 = mm->add_literal(migraphx::literal({migraphx::shape::float_type, {2, 4}}, w));
    auto l0 = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {5, 2}});
    mm->add_instruction(migraphx::make_op("dot"), l0, l1);

    auto prog = optimize_onnx("initializer_not_an_input.onnx");

    EXPECT(p == prog);
}

TEST_CASE(instance_norm_test)
{
    std::vector<size_t> dims{1, 2, 3, 3};
    migraphx::shape s1{migraphx::shape::float_type, dims};
    migraphx::shape s2{migraphx::shape::float_type, {2}};

    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto x     = mm->add_parameter("0", s1);
    auto scale = mm->add_parameter("1", s2);
    auto bias  = mm->add_parameter("2", s2);

    auto mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), x);
    auto mean_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", dims}}), mean);
    auto l0       = mm->add_instruction(migraphx::make_op("sqdiff"), x, mean_bcast);
    auto variance = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), l0);
    auto l1       = mm->add_instruction(migraphx::make_op("sub"), x, mean_bcast);
    auto epsilon_literal = mm->add_literal(1e-5f);
    auto epsilon_bcast   = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", dims}}), epsilon_literal);
    auto variance_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", dims}}), variance);
    auto l2 = mm->add_instruction(migraphx::make_op("add"), variance_bcast, epsilon_bcast);
    auto l3 = mm->add_instruction(migraphx::make_op("rsqrt"), l2);
    auto l4 = mm->add_instruction(migraphx::make_op("mul"), l1, l3);
    auto scale_bcast =
        mm->add_instruction(migraphx::make_op("broadcast", {{"axis", 1}, {"dims", dims}}), scale);
    auto bias_bcast =
        mm->add_instruction(migraphx::make_op("broadcast", {{"axis", 1}, {"dims", dims}}), bias);
    auto l5 = mm->add_instruction(migraphx::make_op("mul"), l4, scale_bcast);
    mm->add_instruction(migraphx::make_op("add"), l5, bias_bcast);

    auto prog = optimize_onnx("instance_norm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(leaky_relu_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    float alpha = 0.01f;
    auto l0     = mm->add_parameter("0", {migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("leaky_relu", {{"alpha", alpha}}), l0);

    auto prog = optimize_onnx("leaky_relu_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(less_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto input1 = mm->add_literal(migraphx::literal(s, data));
    auto input2 = mm->add_parameter("x2", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    auto le     = mm->add_instruction(migraphx::make_op("less"), input1, input2);
    auto ret    = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        le);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("less_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(less_bool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sf{migraphx::shape::float_type, {2, 3}};
    migraphx::shape sb{migraphx::shape::bool_type, {2, 3}};

    auto input1 = mm->add_parameter("x1", sf);
    auto input2 = mm->add_parameter("x2", sb);
    auto cin1   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        input1);
    auto ret = mm->add_instruction(migraphx::make_op("less"), cin1, input2);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("less_bool_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(log_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("log"), input);

    auto prog = optimize_onnx("log_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(logsoftmax_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    int axis = 1;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), l0);
    auto prog = optimize_onnx("logsoftmax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(lrn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 28, 24, 24}});
    migraphx::op::lrn op;
    op.size  = 5;
    op.alpha = 0.0001;
    op.beta  = 0.75;
    op.bias  = 1.0;
    mm->add_instruction(op, l0);
    auto prog = optimize_onnx("lrn_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_bmbm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1 = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {5, 2, 1, 7, 8}});
    auto bl0 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {5, 2, 3, 6, 7}}}), l0);
    auto bl1 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {5, 2, 3, 7, 8}}}), l1);
    mm->add_instruction(migraphx::make_op("dot", {{"alpha", 1.0f}, {"beta", 0.0f}}), bl0, bl1);

    auto prog = optimize_onnx("matmul_bmbm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_bmv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto bsl1 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {3, 7, 1}}}), sl1);
    auto res =
        mm->add_instruction(migraphx::make_op("dot", {{"alpha", 1.0f}, {"beta", 0.0f}}), l0, bsl1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), res);

    auto prog = optimize_onnx("matmul_bmv_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_mv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto res =
        mm->add_instruction(migraphx::make_op("dot", {{"alpha", 1.0f}, {"beta", 0.0f}}), l0, sl1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), res);

    auto prog = optimize_onnx("matmul_mv_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_vbm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {5, 7, 8}});
    auto sl0 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l0);
    auto bsl0 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {5, 1, 7}}}), sl0);
    auto res =
        mm->add_instruction(migraphx::make_op("dot", {{"alpha", 1.0f}, {"beta", 0.0f}}), bsl0, l1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), res);

    auto prog = optimize_onnx("matmul_vbm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_vm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7, 8}});
    auto sl0 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l0);
    auto res =
        mm->add_instruction(migraphx::make_op("dot", {{"alpha", 1.0f}, {"beta", 0.0f}}), sl0, l1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), res);

    auto prog = optimize_onnx("matmul_vm_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmul_vv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl0 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l0);
    auto sl1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto res =
        mm->add_instruction(migraphx::make_op("dot", {{"alpha", 1.0f}, {"beta", 0.0f}}), sl0, sl1);
    auto sr0 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), res);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sr0);

    auto prog = optimize_onnx("matmul_vv_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(matmulinteger_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::int8_type, {3, 6, 16}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::int8_type, {3, 16, 8}});
    mm->add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), l0, l1);

    auto prog = optimize_onnx("matmulinteger_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(max_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = mm->add_instruction(migraphx::make_op("max"), input0, input1);
    mm->add_instruction(migraphx::make_op("max"), l0, input2);

    optimize_onnx("max_test.onnx");
}

TEST_CASE(maxpool_notset_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0, 1, 1};
    float val                 = std::numeric_limits<float>::lowest();
    auto ins_pad =
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}, {"value", val}}), input);
    mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "max"}, {"padding", {0, 0}}, {"stride", {2, 2}}, {"lengths", {6, 6}}}),
        ins_pad);

    auto prog = optimize_onnx("maxpool_notset_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(maxpool_same_upper_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0, 1, 1};
    float val                 = std::numeric_limits<float>::lowest();
    auto ins_pad =
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}, {"value", val}}), input);
    mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "max"}, {"padding", {0, 0}}, {"stride", {1, 1}}, {"lengths", {2, 2}}}),
        ins_pad);

    auto prog = optimize_onnx("maxpool_same_upper_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(min_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = mm->add_instruction(migraphx::make_op("min"), input0, input1);
    mm->add_instruction(migraphx::make_op("min"), l0, input2);

    optimize_onnx("min_test.onnx");
}

TEST_CASE(no_pad_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_onnx("no_pad_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(neg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int64_type, {2, 3}};
    auto input = mm->add_parameter("0", s);
    auto ret   = mm->add_instruction(migraphx::make_op("neg"), input);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("neg_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(nonzero_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data = {1, 0, 1, 1};
    mm->add_literal(migraphx::literal(s, data));

    migraphx::shape si{migraphx::shape::int64_type, {2, 3}};
    std::vector<int64_t> indices = {0, 1, 1, 0, 0, 1};
    auto r                       = mm->add_literal(migraphx::literal(si, indices));
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("nonzero_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(nonzero_int_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int16_type, {2, 3}};
    std::vector<int> data = {1, 1, 0, 1, 0, 1};
    mm->add_literal(migraphx::literal(s, data.begin(), data.end()));

    migraphx::shape si{migraphx::shape::int64_type, {2, 4}};
    std::vector<int64_t> indices = {0, 0, 1, 1, 0, 1, 0, 2};
    auto r                       = mm->add_literal(migraphx::literal(si, indices));
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("nonzero_int_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(onehot_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_ind{migraphx::shape::int32_type, {5, 2}};
    migraphx::shape s_val{migraphx::shape::half_type, {2}};
    mm->add_literal(3);
    auto l_ind = mm->add_parameter("indices", s_ind);
    auto l_val = mm->add_parameter("values", s_val);
    migraphx::shape s_dep{migraphx::shape::half_type, {3, 3}};
    std::vector<float> data_dep{1, 0, 0, 0, 1, 0, 0, 0, 1};
    auto l_dep      = mm->add_literal(migraphx::literal(s_dep, data_dep));
    auto gather_out = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), l_dep, l_ind);
    auto tr_out =
        mm->add_instruction(migraphx::make_op("transpose", {{"dims", {2, 0, 1}}}), gather_out);
    auto off_val = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), l_val);
    auto on_val = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), l_val);
    auto diff       = mm->add_instruction(migraphx::make_op("sub"), on_val, off_val);
    auto mb_off_val = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {3, 5, 2}}}), off_val);
    auto mb_diff = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {3, 5, 2}}}), diff);
    auto mul = mm->add_instruction(migraphx::make_op("mul"), tr_out, mb_diff);
    auto r   = mm->add_instruction(migraphx::make_op("add"), mul, mb_off_val);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("onehot_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pad_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {1, 1, 1, 1}}}), l0);
    auto prog = optimize_onnx("pad_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pad_3arg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    mm->add_literal({migraphx::shape{migraphx::shape::float_type}, {1.0f}});
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 1, 2, 2}});
    auto r = mm->add_instruction(
        migraphx::make_op("pad", {{"pads", {1, 1, 2, 2}}, {"value", 1.0f}}), l0);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("pad_3arg_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pad_reflect_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {0, 2, 0, 1}});
    auto l1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 1}}, {"ends", {2, 2}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 0}}, {"ends", {2, 1}}}), l0);
    auto l3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 0}}, {"ends", {2, 1}}}), l0);
    auto r = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l2, l1, l0, l3);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("pad_reflect_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pad_reflect_multiaxis_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {0, 2, 2, 0}});
    auto l1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 1}}, {"ends", {2, 2}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 2}}, {"ends", {2, 3}}}), l0);
    auto l3 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l2, l1, l0);
    auto l4 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 0}}, {"ends", {1, 5}}}), l3);
    auto l5 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {1, 0}}, {"ends", {2, 5}}}), l3);
    auto r = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), l3, l4, l5);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("pad_reflect_multiaxis_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pow_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    mm->add_instruction(migraphx::make_op("pow"), l0, l1);

    auto prog = optimize_onnx("pow_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pow_fp32_i64_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 5}});
    auto l1f = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l1);
    auto ret = mm->add_instruction(migraphx::make_op("pow"), l0, l1f);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("pow_fp32_i64_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(pow_i64_fp32_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l0f = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l0);
    auto fr = mm->add_instruction(migraphx::make_op("pow"), l0f, l1);
    auto ir = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), fr);
    mm->add_return({ir});

    auto prog = migraphx::parse_onnx("pow_i64_fp32_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(prelu_brcst_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 5}});
    auto bl1 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", l0->get_shape().lens()}}), l1);
    auto ret = mm->add_instruction(migraphx::make_op("prelu"), l0, bl1);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("prelu_brcst_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(range_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(int64_t{10});
    mm->add_literal(int64_t{6});
    mm->add_literal(int64_t{-3});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {2}}, {10, 7}});

    auto prog = optimize_onnx("range_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(range_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(float{2});
    mm->add_literal(float{11});
    mm->add_literal(float{2});
    mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {5}}, {2, 4, 6, 8, 10}});

    auto prog = optimize_onnx("range_float_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(recip_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("recip"), input);

    auto prog = optimize_onnx("recip_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducel1_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto abs_l0 = mm->add_instruction(migraphx::make_op("abs"), l0);
    auto sum_l0 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-2}}}), abs_l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {-2}}}), sum_l0);
    auto prog = optimize_onnx("reducel1_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducel2_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto square_l0 = mm->add_instruction(migraphx::make_op("mul"), l0, l0);
    auto sum_l0 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-1}}}), square_l0);
    auto squ_l0 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {-1}}}), sum_l0);
    mm->add_instruction(migraphx::make_op("sqrt"), squ_l0);
    auto prog = optimize_onnx("reducel2_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reduce_log_sum_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto sum_l0 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-3}}}), l0);
    mm->add_instruction(migraphx::make_op("log"), sum_l0);
    auto prog = optimize_onnx("reduce_log_sum_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reduce_log_sum_exp_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto exp_l0 = mm->add_instruction(migraphx::make_op("exp"), l0);
    auto sum_l0 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-4}}}), exp_l0);
    mm->add_instruction(migraphx::make_op("log"), sum_l0);
    auto prog = optimize_onnx("reduce_log_sum_exp_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducemax_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), l0);
    auto prog = optimize_onnx("reducemax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducemean_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), l1);
    auto prog = optimize_onnx("reducemean_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducemean_keepdims_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), l0);
    auto prog = optimize_onnx("reducemean_keepdims_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducemin_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {2, 3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), l1);
    auto prog = optimize_onnx("reducemin_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reduceprod_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    mm->add_instruction(migraphx::make_op("reduce_prod", {{"axes", {2}}}), l0);
    auto prog = optimize_onnx("reduceprod_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), l1);
    auto prog = optimize_onnx("reducesum_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_multiaxis_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), l1);
    auto prog = optimize_onnx("reducesum_multiaxis_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_keepdims_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), l0);
    auto prog = optimize_onnx("reducesum_keepdims_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reducesum_square_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto squ_l0 = mm->add_instruction(migraphx::make_op("mul"), l0, l0);
    auto sum_l0 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-2}}}), squ_l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {-2}}}), sum_l0);
    auto prog = optimize_onnx("reducesum_square_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reshape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::op::reshape op;
    std::vector<int64_t> reshape_dims{3, 8};
    mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, reshape_dims});
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});
    op.dims = reshape_dims;
    mm->add_instruction(op, l0);
    mm->add_instruction(op, l0);
    auto prog = optimize_onnx("reshape_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(reshape_non_standard_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::op::reshape op;
    std::vector<int64_t> reshape_dims{4, 3, 2};
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
    auto x      = mm->add_parameter("x", s);
    auto tran_x = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 1}}}), x);
    auto cont_x = mm->add_instruction(migraphx::make_op("contiguous"), tran_x);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 3, 2}}}), cont_x);
    auto prog = optimize_onnx("reshape_non_standard_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(resize_downsample_f_test)
{
    migraphx::program p;
    auto* mm              = p.get_main_module();
    std::vector<float> ds = {1.0f, 1.0f, 0.6f, 0.6f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    mm->add_literal(migraphx::literal{ss, ds});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto inx = mm->add_parameter("X", sx);

    mm->add_instruction(migraphx::make_op("undefined"));

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 1, 2}};
    std::vector<int> ind = {4, 7};
    auto li              = mm->add_literal(migraphx::literal(si, ind));

    auto lrsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), inx);
    auto r    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("resize_downsample_f_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(resize_downsample_c_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> ds = {1.0f, 1.0f, 0.6f, 0.6f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    mm->add_literal(migraphx::literal{ss, ds});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto inx = mm->add_parameter("X", sx);

    mm->add_instruction(migraphx::make_op("undefined"));

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 1, 2}};
    std::vector<int> ind = {0, 2};
    auto li              = mm->add_literal(migraphx::literal(si, ind));

    auto lrsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), inx);
    auto r    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("resize_downsample_c_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(resize_outsize_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<int64_t> out_len = {1, 1, 4, 6};
    migraphx::shape so{migraphx::shape::int64_type, {4}};
    mm->add_literal(migraphx::literal(so, out_len));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto inx = mm->add_parameter("X", sx);

    mm->add_instruction(migraphx::make_op("undefined"));

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3};
    auto li              = mm->add_literal(migraphx::literal(si, ind));

    auto lrsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
    auto r    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("resize_outsize_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(resize_upsample_pc_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> ds = {1.0f, 1.0f, 2.0f, 1.5f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    mm->add_literal(migraphx::literal{ss, ds});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto inx = mm->add_parameter("X", sx);

    mm->add_instruction(migraphx::make_op("undefined"));

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 1, 1, 2, 3, 3, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 4, 5, 5, 6, 7, 7};
    auto li              = mm->add_literal(migraphx::literal(si, ind));

    auto lrsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), inx);
    auto r    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("resize_upsample_pc_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(resize_upsample_pf_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> ds = {1.0f, 1.0f, 2.0f, 3.0f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    mm->add_literal(migraphx::literal{ss, ds});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto inx = mm->add_parameter("X", sx);

    mm->add_instruction(migraphx::make_op("undefined"));

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
    auto li              = mm->add_literal(migraphx::literal(si, ind));

    auto lrsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
    auto r    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("resize_upsample_pf_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(round_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::double_type, {10, 5}});
    mm->add_instruction(migraphx::make_op("round"), input);

    auto prog = optimize_onnx("round_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(selu_test)
{
    migraphx::program p;
    auto* mm                      = p.get_main_module();
    std::vector<std::size_t> lens = {2, 3};
    migraphx::shape s{migraphx::shape::double_type, lens};
    auto x = mm->add_parameter("x", s);

    migraphx::shape ls{migraphx::shape::double_type, {1}};
    auto la = mm->add_literal({ls, {0.3}});
    auto lg = mm->add_literal({ls, {0.25}});
    auto mbla =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", lens}}), la);
    auto mblg =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", lens}}), lg);

    auto sign_x = mm->add_instruction(migraphx::make_op("sign"), x);
    auto exp_x  = mm->add_instruction(migraphx::make_op("exp"), x);

    auto mlax  = mm->add_instruction(migraphx::make_op("mul"), mbla, exp_x);
    auto smlax = mm->add_instruction(migraphx::make_op("sub"), mlax, mbla);

    auto item1 = mm->add_instruction(migraphx::make_op("add"), smlax, x);
    auto item2 = mm->add_instruction(migraphx::make_op("sub"), smlax, x);

    auto sitem2 = mm->add_instruction(migraphx::make_op("mul"), sign_x, item2);
    auto item12 = mm->add_instruction(migraphx::make_op("sub"), item1, sitem2);
    auto r      = mm->add_instruction(migraphx::make_op("mul"), item12, mblg);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("selu_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 4, 5, 6}};
    auto l0 = mm->add_parameter("x", s);
    migraphx::shape s_shape{migraphx::shape::int64_type, {4}};
    mm->add_literal(s_shape, l0->get_shape().lens());
    auto prog = optimize_onnx("shape_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(shape_gather_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {7, 3, 10}});
    migraphx::shape const_shape{migraphx::shape::int32_type, {1}};
    auto l2 = mm->add_literal(migraphx::literal{const_shape, {1}});
    auto l1 =
        mm->add_literal(migraphx::shape{migraphx::shape::int64_type, {3}}, l0->get_shape().lens());
    int axis = 0;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), l1, l2);
    auto prog = optimize_onnx("shape_gather_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sign_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::double_type, {10, 5}});
    mm->add_instruction(migraphx::make_op("sign"), input);

    auto prog = optimize_onnx("sign_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sin_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("sin"), input);

    auto prog = optimize_onnx("sin_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sinh_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("sinh"), input);

    auto prog = optimize_onnx("sinh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(slice_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 2}});
    mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {1, 0}}, {"ends", {2, 2}}}), l0);
    auto prog = optimize_onnx("slice_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(slice_3arg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 5}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {0, 0}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {2, 5}});
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 0}}, {"ends", {2, 5}}}), l0);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("slice_3arg_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(slice_5arg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 5}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {1, 1}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-1, -2}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-1, -1}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-5, -3}});
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {-1, -2}}, {"starts", {-5, -3}}, {"ends", {-1, -1}}}),
        l0);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("slice_5arg_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(slice_max_end_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {10, 20}});
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {0, 1}}, {"starts", {1, 2}}, {"ends", {3000000000, -1}}}),
        l0);
    auto prog = optimize_onnx("slice_max_end_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(softmax_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3}});
    mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), l0);
    auto prog = optimize_onnx("softmax_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(split_minus_axis_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    auto r1    = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {-1}}, {"starts", {0}}, {"ends", {5}}}), input);
    auto r2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {-1}}, {"starts", {5}}, {"ends", {10}}}), input);
    auto r3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {-1}}, {"starts", {10}}, {"ends", {15}}}), input);
    mm->add_return({r1, r2, r3});

    auto prog = migraphx::parse_onnx("split_minus_axis_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(split_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    auto r1    = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {7}}}), input);
    auto r2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {7}}, {"ends", {11}}}), input);
    auto r3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {11}}, {"ends", {15}}}), input);
    mm->add_return({r1, r2, r3});

    auto prog = migraphx::parse_onnx("split_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(split_test_default)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    auto r1    = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {5}}}), input);
    auto r2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {5}}, {"ends", {10}}}), input);
    mm->add_return({r1, r2});

    auto prog = migraphx::parse_onnx("split_test_default.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sqrt_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10, 15}});
    mm->add_instruction(migraphx::make_op("sqrt"), input);

    auto prog = optimize_onnx("sqrt_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(squeeze_unsqueeze_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int64_t> squeeze_axes{0, 2, 3, 5};
    std::vector<int64_t> unsqueeze_axes{0, 1, 3, 5};
    auto l0 =
        mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 1, 1, 2, 1}});
    auto l1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", squeeze_axes}}), l0);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsqueeze_axes}}), l1);
    auto prog = optimize_onnx("squeeze_unsqueeze_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sub_bcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2  = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"dims", l0->get_shape().lens()}}), l1);
    mm->add_instruction(migraphx::make_op("sub"), l0, l2);

    auto prog = optimize_onnx("sub_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sub_scalar_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1}});
    auto m1 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {2, 3, 4, 5}}}), l1);
    mm->add_instruction(migraphx::make_op("sub"), l0, m1);
    auto prog = optimize_onnx("sub_scalar_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sum_int_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::int16_type, {3}});
    auto input1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::uint16_type, {3}});
    auto input2 = mm->add_parameter("2", migraphx::shape{migraphx::shape::uint32_type, {3}});
    auto cin0   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::uint32_type)}}),
        input0);
    auto cin1 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::uint32_type)}}),
        input1);
    auto l0 = mm->add_instruction(migraphx::make_op("add"), cin0, cin1);
    mm->add_instruction(migraphx::make_op("add"), l0, input2);

    auto prog = optimize_onnx("sum_int_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sum_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto l0     = mm->add_instruction(migraphx::make_op("add"), input0, input1);
    mm->add_instruction(migraphx::make_op("add"), l0, input2);

    auto prog = optimize_onnx("sum_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(sum_type_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l_bool   = mm->add_literal({migraphx::shape{migraphx::shape::bool_type, {2}}, {1, 0}});
    auto l_int8   = mm->add_literal({migraphx::shape{migraphx::shape::int8_type, {2}}, {1, 1}});
    auto l_uint8  = mm->add_literal({migraphx::shape{migraphx::shape::uint8_type, {2}}, {1, 1}});
    auto l_uint16 = mm->add_literal({migraphx::shape{migraphx::shape::uint16_type, {2}}, {1, 1}});
    auto l_uint32 = mm->add_literal({migraphx::shape{migraphx::shape::uint32_type, {2}}, {1, 1}});
    auto l_uint64 = mm->add_literal({migraphx::shape{migraphx::shape::uint64_type, {2}}, {1, 1}});
    auto l_double = mm->add_literal({migraphx::shape{migraphx::shape::double_type, {2}}, {1, 1}});
    auto l_raw  = mm->add_literal({migraphx::shape{migraphx::shape::double_type, {2}}, {1.5, 2.0}});
    auto o_bool = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_bool);
    auto o_int8 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_int8);
    auto o_uint8 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_uint8);
    auto o_uint16 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_uint16);
    auto o_uint32 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_uint32);
    auto o_uint64 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_uint64);
    auto s0 = mm->add_instruction(migraphx::make_op("add"), o_bool, o_int8);
    auto s1 = mm->add_instruction(migraphx::make_op("add"), s0, o_uint8);
    auto s2 = mm->add_instruction(migraphx::make_op("add"), s1, o_uint16);
    auto s3 = mm->add_instruction(migraphx::make_op("add"), s2, o_uint32);
    auto s4 = mm->add_instruction(migraphx::make_op("add"), s3, o_uint64);
    auto s5 = mm->add_instruction(migraphx::make_op("add"), s4, l_double);
    auto s6 = mm->add_instruction(migraphx::make_op("add"), s5, l_raw);
    mm->add_return({s6});

    auto prog = migraphx::parse_onnx("sum_type_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(tan_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(migraphx::make_op("tan"), input);

    auto prog = optimize_onnx("tan_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(tanh_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    mm->add_instruction(migraphx::make_op("tanh"), input);

    auto prog = optimize_onnx("tanh_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(tile_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, {1, 2}});
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), input, input);

    auto prog = optimize_onnx("tile_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(tile_test_3x2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, {3, 2}});
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    auto l0    = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), input, input);
    auto l1    = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), l0, input);
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l1, l1);

    auto prog = optimize_onnx("tile_test_3x2.onnx");

    EXPECT(p == prog);
}

TEST_CASE(transpose_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    std::vector<int64_t> perm{0, 3, 1, 2};
    mm->add_instruction(migraphx::make_op("transpose", {{"dims", perm}}), input);

    auto prog = optimize_onnx("transpose_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(transpose_gather_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    auto make_contiguous = [&mm](migraphx::instruction_ref ins) {
        if(ins->get_shape().standard())
        {
            return ins;
        }

        return mm->add_instruction(migraphx::make_op("contiguous"), ins);
    };

    auto data =
        mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 5, 4, 6}});
    auto ind =
        mm->add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, {2, 4, 3, 5}});
    auto tr_data =
        mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 1, 3}}}), data);
    auto tr_ind =
        mm->add_instruction(migraphx::make_op("transpose", {{"dims", {0, 2, 1, 3}}}), ind);
    int axis = 1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}),
                        make_contiguous(tr_data),
                        make_contiguous(tr_ind));

    auto prog = optimize_onnx("transpose_gather_test.onnx");

    EXPECT(p.sort() == prog.sort());
}

TEST_CASE(undefined_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = mm->add_instruction(migraphx::make_op("undefined"));
    auto l2 = mm->add_instruction(migraphx::make_op("identity"), l1);
    mm->add_return({l2});

    auto prog = migraphx::parse_onnx("undefined_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(unknown_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto l2  = mm->add_instruction(migraphx::op::unknown{"Unknown"}, l0, l1);
    mm->add_instruction(migraphx::op::unknown{"Unknown"}, l2);
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

TEST_CASE(upsample_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    mm->add_literal(migraphx::literal(ss, {1.0f, 1.0f, 2.0f, 3.0f}));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto ix = mm->add_parameter("X", sx);

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};

    auto li  = mm->add_literal(migraphx::literal(si, ind));
    auto rsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), ix);
    auto r   = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp, li);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("upsample_test.onnx");

    EXPECT(p == prog);
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
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_onnx("variable_batch_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 16, 16}});
    auto r   = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.default_dim_value = 2;

    auto prog = migraphx::parse_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_leq_zero_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    auto prog = optimize_onnx("variable_batch_leq_zero_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(where_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto lc  = mm->add_parameter("c", migraphx::shape{migraphx::shape::bool_type, {2}});
    auto lx  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2, 2}});
    auto ly  = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 1, 2, 2}});

    auto int_c = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::int32_type)}}),
        lc);
    auto lccm = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {2, 2, 2, 2}}}), int_c);
    auto lxm = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {2, 2, 2, 2}}}), lx);
    auto lym = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", {2, 2, 2, 2}}}), ly);

    auto concat_data = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), lym, lxm);
    auto rsp_data =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {32}}}), concat_data);

    std::vector<int> offset(16, 16);
    std::vector<int> ind(16);
    std::iota(ind.begin(), ind.end(), 0);
    migraphx::shape ind_s{migraphx::shape::int32_type, {2, 2, 2, 2}};

    auto lind    = mm->add_literal(migraphx::literal(ind_s, ind));
    auto loffset = mm->add_literal(migraphx::literal(ind_s, offset));

    auto ins_co  = mm->add_instruction(migraphx::make_op("mul"), loffset, lccm);
    auto ins_ind = mm->add_instruction(migraphx::make_op("add"), ins_co, lind);
    auto r = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp_data, ins_ind);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("where_test.onnx");

    EXPECT(p == prog);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
