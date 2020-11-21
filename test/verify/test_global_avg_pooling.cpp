
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>

struct test_global_avg_pooling : verify_program<test_global_avg_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
        auto op    = migraphx::op::pooling{"average"};
        auto lens  = input->get_shape().lens();
        op.lengths = {lens[2], lens[3]};
        mm->add_instruction(op, input);
        return p;
    }
};
