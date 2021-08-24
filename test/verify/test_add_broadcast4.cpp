
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/instruction.hpp>

struct test_add_broadcast4 : verify_program<test_add_broadcast4>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x  = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3, 5}});
        auto y  = mm->add_parameter("y", {migraphx::shape::float_type, {3}});
        auto by = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", x->get_shape().lens()}}), y);
        mm->add_instruction(migraphx::make_op("add"), x, by);
        return p;
    }
};
