
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/op/pooling.hpp>

struct test_avg_pooling_3d : verify_program<test_avg_pooling_3d>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 3, 5, 5, 5}});
        auto op = migraphx::op::pooling{"average", {1, 1, 1}, {3, 3, 3}, {3, 3, 3}};
        mm->add_instruction(op, input);
        return p;
    }
};
