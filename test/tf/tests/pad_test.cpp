
#include <tf_test.hpp>

TEST_CASE(pad_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4}});
    std::vector<int> pad_literals{1, 1, 2, 2};
    std::vector<int> pads{1, 2, 1, 2};
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {2, 2}}, pad_literals);

    mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), l0);
    auto prog = optimize_tf("pad_test.pb", false);

    EXPECT(p == prog);
}


