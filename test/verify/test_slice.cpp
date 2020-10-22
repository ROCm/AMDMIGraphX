
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_slice : verify_program<test_slice>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::int32_type, {2, 2, 4}};
        auto x      = p.add_parameter("x", s);
        auto y      = p.add_parameter("y", {migraphx::shape::int32_type, {2, 2, 2}});
        auto slice0 = p.add_instruction(migraphx::op::slice{{2}, {0}, {2}}, x);
        p.add_instruction(migraphx::op::add{}, y, slice0);

        return p;
    }
};


