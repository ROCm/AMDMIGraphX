
#include <tf_test.hpp>

TEST_CASE(pooling_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    migraphx::op::pooling avg_pool_op{migraphx::op::pooling_mode::average};
    migraphx::op::pooling max_pool_op{migraphx::op::pooling_mode::max};
    avg_pool_op.stride  = {2, 2};
    max_pool_op.stride  = {2, 2};
    avg_pool_op.lengths = {2, 2};
    max_pool_op.lengths = {2, 2};
    mm->add_instruction(avg_pool_op, l0);
    mm->add_instruction(max_pool_op, l0);
    auto prog = optimize_tf("pooling_test.pb", true);

    EXPECT(p == prog);
}


