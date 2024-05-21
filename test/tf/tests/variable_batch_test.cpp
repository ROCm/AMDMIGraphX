
#include <tf_test.hpp>

TEST_CASE(variable_batch_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_tf("variable_batch_test.pb", false);

    EXPECT(p == prog);
}


