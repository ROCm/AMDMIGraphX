
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <int Axis, migraphx::shape::type_t T>
struct test_logsoftmax : verify_program<test_logsoftmax<Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{T, {10, 4, 2080, 6}};
        auto param = mm->add_parameter("0", s);
        mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", Axis}}), param);

        return p;
    }
};

template struct test_logsoftmax<0, migraphx::shape::float_type>;
template struct test_logsoftmax<1, migraphx::shape::float_type>;
template struct test_logsoftmax<2, migraphx::shape::float_type>;
template struct test_logsoftmax<3, migraphx::shape::float_type>;
template struct test_logsoftmax<1, migraphx::shape::half_type>;
template struct test_logsoftmax<0, migraphx::shape::half_type>;
template struct test_logsoftmax<2, migraphx::shape::half_type>;
template struct test_logsoftmax<3, migraphx::shape::half_type>;
