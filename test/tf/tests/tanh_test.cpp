
#include <tf_test.hpp>

TEST_CASE(tanh_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto l1  = mm->add_instruction(migraphx::make_op("tanh"), l0);
    mm->add_return({l1});
    auto prog = parse_tf("tanh_test.pb", false);

    EXPECT(p == prog);
}


