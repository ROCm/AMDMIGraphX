
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/quantization.hpp>

struct test_fp32_fp16_sub : verify_program<test_fp32_fp16_sub>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1   = p.add_parameter("x", s);
        auto p2   = p.add_parameter("y", s);
        auto sum  = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto diff = p.add_instruction(migraphx::op::sub{}, sum, p2);
        p.add_instruction(migraphx::op::add{}, diff, p1);
        migraphx::quantize_fp16(p, {"sub"});

        return p;
    };
};


