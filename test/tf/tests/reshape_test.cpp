
#include <tf_test.hpp>

TEST_CASE(reshape_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {16}});
    migraphx::shape s0{migraphx::shape::int32_type, {4}};
    // in tf, the second arg is a literal that contains new dimensions
    mm->add_literal(migraphx::literal{s0, {1, 1, 1, 16}});
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1, 16}}}), l0);
    auto prog = optimize_tf("reshape_test.pb", false);

    EXPECT(p == prog);
}


