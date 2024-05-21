
#include <tf_test.hpp>

TEST_CASE(stridedslice_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 10, 1, 1}});
    auto l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {0, 0, 0, 0}}, {"ends", {1, 1, 1, 5}}, {"axes", {0, 1, 2, 3}}}),
        l1);
    auto shrink_axis = 1;
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {shrink_axis}}}), l2);
    auto prog = optimize_tf("stridedslice_test.pb", true);

    EXPECT(p == prog);
}


