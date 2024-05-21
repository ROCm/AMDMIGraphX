
#include <tf_test.hpp>

TEST_CASE(relu6_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    std::vector<size_t> input_lens{1, 3, 16, 16};
    auto l0      = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, input_lens});
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  min_val);
    max_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  max_val);
    mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    auto prog = optimize_tf("relu6_test.pb", false);

    EXPECT(p == prog);
}


