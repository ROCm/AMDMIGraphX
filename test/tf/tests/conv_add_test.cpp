
#include <tf_test.hpp>
#include <tf_conv_utils.hpp>


TEST_CASE(conv_add_test)
{
    migraphx::program p = create_conv();
    auto* mm            = p.get_main_module();
    auto l0             = std::prev(mm->end());
    mm->add_instruction(migraphx::make_op("add"), l0, l0);
    auto prog = optimize_tf("conv_add_test.pb", true);

    EXPECT(p == prog);
}


