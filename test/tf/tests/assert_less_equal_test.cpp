
#include <tf_test.hpp>


TEST_CASE(assert_less_equal_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {2, 3}};
    auto l0 = mm->add_parameter("0", s0);
    auto l1 = mm->add_parameter("1", s0);
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {0, 1}};
    auto l2 = mm->add_literal(l);
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    auto l3 = mm->add_instruction(migraphx::make_op("identity"), l0, l1);
    mm->add_instruction(migraphx::make_op("identity"), l3, l2);
    auto prog = optimize_tf("assert_less_equal_test.pb", false);

    EXPECT(p == prog);
}


