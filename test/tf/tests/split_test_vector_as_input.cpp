
#include <tf_test.hpp>

TEST_CASE(split_test_vector_as_input)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    std::vector<int64_t> axes{0, 1};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    // split sizes
    mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {3}}, {4, 15, 11}});
    mm->add_literal(1); // split axis
    mm->add_literal(1); // concat axis
    mm->add_literal(1); // concat axis
    auto l1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 0}}, {"ends", {5, 4}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 4}}, {"ends", {5, 19}}}), l0);
    auto l3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 19}}, {"ends", {5, 30}}}), l0);
    auto l4 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l1, l2);
    auto l5 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l2, l3);
    mm->add_return({l4, l5});
    auto prog = parse_tf("split_test_vector_as_input.pb", false);

    EXPECT(p == prog);
}


