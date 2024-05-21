
#include <tf_test.hpp>


TEST_CASE(biasadd_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {1, 500, 1, 1}};
    uint64_t axis = 1;
    auto l0       = mm->add_parameter("0", s0);
    auto l1       = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {500}});
    auto l2       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l0->get_shape().lens()}}), l1);
    mm->add_instruction(migraphx::make_op("add"), l0, l2);
    auto prog = optimize_tf("biasadd_test.pb", true);

    EXPECT(p == prog);
}


