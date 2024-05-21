
#include <tf_test.hpp>
#include <tf_conv_utils.hpp>


TEST_CASE(conv_relu6_test)
{
    migraphx::program p = create_conv();
    auto* mm            = p.get_main_module();
    std::vector<size_t> input_lens{1, 32, 16, 16};
    auto l0      = std::prev(mm->end());
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  min_val);
    max_val = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                  max_val);
    mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    auto prog = optimize_tf("conv_relu6_test.pb", true);

    EXPECT(p == prog);
}


