
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_slice_sin : verify_program<test_slice_sin>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto l   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto t   = mm->add_instruction(migraphx::op::slice{{1}, {1}, {2}}, l);
        mm->add_instruction(migraphx::op::sin{}, t);

        return p;
    }
};
