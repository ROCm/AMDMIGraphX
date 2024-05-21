
#include <tf_test.hpp>


TEST_CASE(argmax_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 5, 6, 7}});
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {2}});
    auto ins = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), l0);
    auto l1  = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), ins);
    mm->add_return({l1});
    auto prog = parse_tf("argmax_test.pb", false, {{"0", {4, 5, 6, 7}}});

    EXPECT(p == prog);
}


