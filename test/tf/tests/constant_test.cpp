
#include <tf_test.hpp>


TEST_CASE(const_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::shape{migraphx::shape::float_type}, std::vector<float>{1.0f});
    auto prog = optimize_tf("constant_test.pb", false);

    EXPECT(p == prog);
}


