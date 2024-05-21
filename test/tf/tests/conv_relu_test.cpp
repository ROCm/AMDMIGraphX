
#include <tf_test.hpp>
#include <conv_test_utils.hpp>


TEST_CASE(conv_relu_test)
{
    migraphx::program p = create_conv();
    auto* mm            = p.get_main_module();
    auto l0             = std::prev(mm->end());
    mm->add_instruction(migraphx::make_op("relu"), l0);
    auto prog = optimize_tf("conv_relu_test.pb", true);

    EXPECT(p == prog);
}


