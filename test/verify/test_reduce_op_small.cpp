
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/op/reduce_sum.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/reduce_max.hpp>
#include <migraphx/op/reduce_min.hpp>
#include <migraphx/op/reduce_prod.hpp>

template <class Op, int Axis, migraphx::shape::type_t T>
struct test_reduce_op_small : verify_program<test_reduce_op_small<Op, Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{T, {3, 4, 2, 2}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(Op{{Axis}}, x);
        return p;
    };
};

template struct test_reduce_op_small<migraphx::op::reduce_sum, 1, migraphx::shape::float_type>;
template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min, 2, migraphx::shape::int32_type>;

template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_prod, -2, migraphx::shape::half_type>;
