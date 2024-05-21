
#include <tf_test.hpp>

TEST_CASE(transpose_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    migraphx::shape s0{migraphx::shape::int32_type, {4}};
    mm->add_literal(migraphx::literal{s0, {0, 2, 3, 1}});
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto prog = optimize_tf("transpose_test.pb", false);

    EXPECT(p == prog);
}


