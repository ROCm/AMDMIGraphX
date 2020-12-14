
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/quantization.hpp>

struct test_fp32_fp16_add : verify_program<test_fp32_fp16_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1   = mm->add_parameter("x", s);
        auto p2   = mm->add_parameter("y", s);
        auto sum  = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        auto diff = mm->add_instruction(migraphx::make_op("sub"), sum, p2);
        mm->add_instruction(migraphx::make_op("add"), diff, p1);
        migraphx::quantize_fp16(p, {"add"});

        return p;
    };
};
