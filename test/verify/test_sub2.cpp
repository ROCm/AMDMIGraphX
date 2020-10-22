
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_sub2 : verify_program<test_sub2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        migraphx::shape b{migraphx::shape::float_type, {3}};
        auto x    = p.add_parameter("x", s);
        auto y    = p.add_parameter("y", s);
        auto z    = p.add_parameter("z", b);
        auto zb   = p.add_instruction(migraphx::op::broadcast{1, s.lens()}, z);
        auto diff = p.add_instruction(migraphx::op::sub{}, x, y);
        p.add_instruction(migraphx::op::sub{}, diff, zb);
        return p;
    }
};


