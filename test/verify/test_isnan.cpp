#include <limits>
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_abs : verify_program<test_abs>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        std::vector<float> data0{3, std::numeric_limits<float>::quiet_NaN()};
        migraphx::shape s0{migraphx::shape::float_type, {1, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s0, data0});
        x       = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, l0);
        mm->add_instruction(migraphx::make_op("isnan"), x);
        return p;
    }
};
