
#include <tf_test.hpp>

TEST_CASE(shape_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 3, 16, 16}});
    auto prog = optimize_tf("shape_test.pb", false);

    EXPECT(p == prog);
}


