
#include <onnx_test.hpp>


TEST_CASE(where_dyn_test)
{
    // TODO: broadcasting for dynamic shapes isn't implemented at time of writing.
    // Update this test case to use shapes that require broadcasting, when available.
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto lc  = mm->add_parameter(
        "c", migraphx::shape{migraphx::shape::bool_type, {{1, 4}, {2, 2}, {2, 2}}});
    auto lx = mm->add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}});
    auto ly = mm->add_parameter(
        "y", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}});

    auto r = mm->add_instruction(migraphx::make_op("where"), lc, lx, ly);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = parse_onnx("where_dyn_test.onnx", options);

    EXPECT(p == prog);
}


