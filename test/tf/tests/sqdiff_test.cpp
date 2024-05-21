
#include <tf_test.hpp>

TEST_CASE(sqdiff_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    mm->add_instruction(migraphx::make_op("sqdiff"), l0, l1);
    auto prog = optimize_tf("sqdiff_test.pb", false);

    EXPECT(p == prog);
}


