
#include <tf_test.hpp>


TEST_CASE(slice_test)
{
    migraphx::program p;

    auto* mm             = p.get_main_module();
    std::size_t num_axes = 2;
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 10}});
    migraphx::shape s0{migraphx::shape::int32_type, {num_axes}};
    mm->add_literal(migraphx::literal{s0, {1, 0}});
    mm->add_literal(migraphx::literal{s0, {2, -1}});

    mm->add_instruction(
        migraphx::make_op("slice", {{"starts", {1, 0}}, {"ends", {3, 10}}, {"axes", {0, 1}}}), l0);
    auto prog = optimize_tf("slice_test.pb", false);

    EXPECT(p == prog);
}


