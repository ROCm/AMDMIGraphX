
#include <onnx_test.hpp>
#include <migraphx/op/unknown.hpp>

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
