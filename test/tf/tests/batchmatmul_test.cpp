
#include <tf_test.hpp>


TEST_CASE(batchmatmul_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 8, 4}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 4, 8}});

    auto trans_l0 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l0);
    auto trans_l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l1);

    mm->add_instruction(migraphx::make_op("dot"), trans_l0, trans_l1);
    auto prog = optimize_tf("batchmatmul_test.pb", false);

    EXPECT(p == prog);
}


