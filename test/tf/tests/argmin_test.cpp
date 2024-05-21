
#include <tf_test.hpp>

TEST_CASE(argmin_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {2}});
    auto ins = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 2}}), l0);
    auto l1  = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), ins);
    mm->add_return({l1});
    auto prog = parse_tf("argmin_test.pb", false);

    EXPECT(p == prog);
}


