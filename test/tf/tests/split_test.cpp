
#include <tf_test.hpp>

TEST_CASE(split_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    std::vector<int64_t> axes{0, 1};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    mm->add_literal(3); // num_splits
    mm->add_literal(1); // split axis
    mm->add_literal(1); // concat axis
    mm->add_literal(1); // concat axis
    auto l1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 0}}, {"ends", {5, 10}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 10}}, {"ends", {5, 20}}}), l0);
    auto l3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 20}}, {"ends", {5, 30}}}), l0);
    auto l4 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l1, l2);
    auto l5 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l2, l3);
    mm->add_return({l4, l5});
    auto prog = parse_tf("split_test.pb", false);

    EXPECT(p == prog);
}


