
#include <tf_test.hpp>


TEST_CASE(onehot_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {5}}, {1, 1, 1, 1, 1}});
    mm->add_literal(2);
    mm->add_literal(1.0f);
    mm->add_literal(0.0f);
    auto l1 = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2, 2}}, {1, 0, 0, 1}});
    int axis = 0;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), l1, l0);
    auto prog = optimize_tf("onehot_test.pb", false);

    EXPECT(p == prog);
}


