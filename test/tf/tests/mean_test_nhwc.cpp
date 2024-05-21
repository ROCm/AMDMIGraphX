
#include <tf_test.hpp>


TEST_CASE(mean_test_nhwc)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {1, 2}};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    migraphx::op::reduce_mean op{{1, 2}};
    auto l2 = mm->add_instruction(op, l1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 2}}}), l2);
    auto prog = optimize_tf("mean_test_nhwc.pb", true);

    EXPECT(p == prog);
}


