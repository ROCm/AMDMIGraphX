
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_add : verify_program<test_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", s);
        mm->add_instruction(migraphx::op::add{}, x, y);
        return p;
    }
};
