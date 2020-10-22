
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>

struct test_triadd_broadcast : verify_program<test_triadd_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x   = p.add_parameter("x", {migraphx::shape::float_type, {2, 2, 3}});
        auto y   = p.add_parameter("y", {migraphx::shape::float_type, {2, 2}});
        auto z   = p.add_parameter("z", {migraphx::shape::float_type, {2, 2, 3}});
        auto by  = p.add_instruction(migraphx::op::broadcast{0, x->get_shape().lens()}, y);
        auto sum = p.add_instruction(migraphx::op::add{}, x, by);
        p.add_instruction(migraphx::op::add{}, sum, z);
        return p;
    }
};


