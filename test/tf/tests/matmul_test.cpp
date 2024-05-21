
#include <tf_test.hpp>

TEST_CASE(matmul_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {8, 4}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 8}});

    auto trans_l0 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l0);
    auto trans_l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);

    mm->add_instruction(migraphx::make_op("dot"), trans_l0, trans_l1);
    auto prog = optimize_tf("matmul_test.pb", false);

    EXPECT(p == prog);
}


