
#include <tf_test.hpp>


TEST_CASE(split_test_one_output)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    mm->add_literal(1); // num_splits
    mm->add_literal(1); // split axis
    auto l1 = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({l1});
    auto prog = parse_tf("split_test_one_output.pb", false);

    EXPECT(p == prog);
}


