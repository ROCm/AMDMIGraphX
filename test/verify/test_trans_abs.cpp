
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_trans_abs : verify_program<test_trans_abs>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto tx = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, x);
        auto absx = p.add_instruction(migraphx::op::abs{}, tx);
        auto r    = p.add_instruction(migraphx::op::add{}, absx, absx);
        p.add_instruction(migraphx::op::contiguous{}, r);

        return p;
    }
};
