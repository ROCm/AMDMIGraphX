
#include <tf_test.hpp>


TEST_CASE(gather_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4}});
    auto l1 = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {2}}, {1, 1}});
    mm->add_literal(1);

    int axis = 1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), l0, l1);
    auto prog = optimize_tf("gather_test.pb", false);

    EXPECT(p == prog);
}


