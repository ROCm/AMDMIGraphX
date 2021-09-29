
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/op/reduce_max.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/reduce_min.hpp>
#include <migraphx/op/reduce_prod.hpp>
#include <migraphx/op/reduce_sum.hpp>

template <class Op, int Axis, migraphx::shape::type_t T>
struct test_reduce_op_large : verify_program<test_reduce_op_large<Op, Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{T, {3, 1026, 4, 3}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(Op{{1}}, x);
        return p;
    };
};

template struct test_reduce_op_large<migraphx::op::reduce_max, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_mean, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_min, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_prod, 2, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_sum, 1, migraphx::shape::float_type>;

struct test_reduce_mean : verify_program<test_reduce_mean>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 384, 1024}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(migraphx::op::reduce_mean{{1}}, x);
        return p;
    };
};
