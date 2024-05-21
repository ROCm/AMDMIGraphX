
#include <tf_test.hpp>


TEST_CASE(add_bcast_test)
{

    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {2, 3}};
    auto l0 = mm->add_parameter("0", s0);
    auto l1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 1}});
    auto l2 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s0.lens()}}), l1);
    mm->add_instruction(migraphx::make_op("add"), l0, l2);
    auto prog = optimize_tf("add_bcast_test.pb", false);

    EXPECT(p == prog);
}


