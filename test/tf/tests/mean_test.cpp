
#include <tf_test.hpp>

TEST_CASE(mean_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {2, 3}};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_literal(l);
    mm->add_literal(l);
    migraphx::op::reduce_mean op{{2, 3}};
    mm->add_instruction(op, l0);
    auto l3 = mm->add_instruction(op, l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), l3);
    auto prog = optimize_tf("mean_test.pb", false);

    EXPECT(p == prog);
}


