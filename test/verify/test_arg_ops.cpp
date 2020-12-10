
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/op/argmax.hpp>
#include <migraphx/op/argmin.hpp>

template <class T, int Axis>
struct test_arg_ops : verify_program<test_arg_ops<T, Axis>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 1025}};
        auto param = mm->add_parameter("data", s);
        mm->add_instruction(T{Axis}, param);

        return p;
    }
};

template struct test_arg_ops<migraphx::op::argmax, 0>;
template struct test_arg_ops<migraphx::op::argmax, 1>;
template struct test_arg_ops<migraphx::op::argmax, 2>;
template struct test_arg_ops<migraphx::op::argmax, 3>;
template struct test_arg_ops<migraphx::op::argmax, -1>;
template struct test_arg_ops<migraphx::op::argmax, -2>;

template struct test_arg_ops<migraphx::op::argmin, 0>;
template struct test_arg_ops<migraphx::op::argmin, 1>;
template struct test_arg_ops<migraphx::op::argmin, 2>;
template struct test_arg_ops<migraphx::op::argmin, 3>;
template struct test_arg_ops<migraphx::op::argmin, -3>;
template struct test_arg_ops<migraphx::op::argmin, -4>;
