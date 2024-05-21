
#include <tf_test.hpp>


TEST_CASE(expanddims_test_neg_dims)
{
    // this check makes sure the pb parses negative dim value correctly
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});
    mm->add_literal(-1);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 4, 1}}}), l0);
    auto prog = optimize_tf("expanddims_neg_test.pb", false);

    EXPECT(p == prog);
}


