
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_mul_add : verify_program<test_mul_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        migraphx::shape bs{migraphx::shape::float_type, {3}};
        auto x  = mm->add_parameter("x", s);
        auto a  = mm->add_parameter("a", bs);
        auto b  = mm->add_parameter("b", bs);
        auto ab = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s.lens()}}), a);
        auto bb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s.lens()}}), b);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), x, ab);
        mm->add_instruction(migraphx::make_op("add"), mul, bb);
        return p;
    }
};
