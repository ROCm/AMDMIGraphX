
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>

struct test_add_broadcast5 : verify_program<test_add_broadcast5>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraphx::shape::float_type, {2, 4, 8}});
        auto y  = p.add_parameter("y", {migraphx::shape::float_type, {4}});
        auto by = p.add_instruction(migraphx::op::broadcast{1, x->get_shape().lens()}, y);
        p.add_instruction(migraphx::op::add{}, x, by);
        return p;
    }
};


