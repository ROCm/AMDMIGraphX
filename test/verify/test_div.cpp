
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_div : verify_program<test_div>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto diff = mm->add_instruction(migraphx::op::div{}, x, y);
        mm->add_instruction(migraphx::op::div{}, diff, z);
        return p;
    }
};
