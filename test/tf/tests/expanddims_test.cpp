
#include <tf_test.hpp>

TEST_CASE(expanddims_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});
    mm->add_literal(0);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 3, 4}}}), l0);
    auto prog = optimize_tf("expanddims_test.pb", false);

    EXPECT(p == prog);
}


