
#include <tf_test.hpp>

TEST_CASE(concat_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 7, 3}});
    auto l1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});

    int axis = 1;
    // tf uses axis as the third input, and it is in int32 format
    // add the literal using a vector in order to set stride to 1 (like in tf parser)
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type}, std::vector<int>{axis});

    mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1);
    auto prog = optimize_tf("concat_test.pb", false);

    EXPECT(p == prog);
}


