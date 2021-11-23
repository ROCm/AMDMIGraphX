
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/argmax.hpp>
#include <migraphx/op/argmin.hpp>

template <class T, int Axis, bool Transpose>
struct test_arg_ops : verify_program<test_arg_ops<T, Axis, Transpose>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 1025}};
        auto param       = mm->add_parameter("data", s);
        if(Transpose) { // test non-standard shape for the arg ops
            param = mm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), param);
        }
        mm->add_instruction(T{Axis}, param);

        return p;
    }
};

template struct test_arg_ops<migraphx::op::argmax, 0, true>;
template struct test_arg_ops<migraphx::op::argmax, 1, true>;
template struct test_arg_ops<migraphx::op::argmax, 2, true>;
template struct test_arg_ops<migraphx::op::argmax, 3, true>;
template struct test_arg_ops<migraphx::op::argmax, -1, true>;
template struct test_arg_ops<migraphx::op::argmax, -2, true>;

template struct test_arg_ops<migraphx::op::argmin, 0, true>;
template struct test_arg_ops<migraphx::op::argmin, 1, true>;
template struct test_arg_ops<migraphx::op::argmin, 2, true>;
template struct test_arg_ops<migraphx::op::argmin, 3, true>;
template struct test_arg_ops<migraphx::op::argmin, -3, true>;
template struct test_arg_ops<migraphx::op::argmin, -4, true>;

template struct test_arg_ops<migraphx::op::argmax, 0, false>;
template struct test_arg_ops<migraphx::op::argmax, 1, false>;
template struct test_arg_ops<migraphx::op::argmax, 2, false>;
template struct test_arg_ops<migraphx::op::argmax, 3, false>;
template struct test_arg_ops<migraphx::op::argmax, -1, false>;
template struct test_arg_ops<migraphx::op::argmax, -2, false>;

template struct test_arg_ops<migraphx::op::argmin, 0, false>;
template struct test_arg_ops<migraphx::op::argmin, 1, false>;
template struct test_arg_ops<migraphx::op::argmin, 2, false>;
template struct test_arg_ops<migraphx::op::argmin, 3, false>;
template struct test_arg_ops<migraphx::op::argmin, -3, false>;
template struct test_arg_ops<migraphx::op::argmin, -4, false>;
