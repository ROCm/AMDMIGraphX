
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/instruction.hpp>

struct test_triadd_broadcast : verify_program<test_triadd_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x  = mm->add_parameter("x", {migraphx::shape::float_type, {2, 2, 3}});
        auto y  = mm->add_parameter("y", {migraphx::shape::float_type, {2, 2}});
        auto z  = mm->add_parameter("z", {migraphx::shape::float_type, {2, 2, 3}});
        auto by = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", x->get_shape().lens()}}), y);
        auto sum = mm->add_instruction(migraphx::make_op("add"), x, by);
        mm->add_instruction(migraphx::make_op("add"), sum, z);
        return p;
    }
};
