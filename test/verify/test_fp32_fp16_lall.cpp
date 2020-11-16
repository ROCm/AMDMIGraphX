
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/quantization.hpp>

struct test_fp32_fp16_lall : verify_program<test_fp32_fp16_lall>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = mm->add_literal(migraphx::literal(s, data));
        auto l2 = mm->add_parameter("p2", s);
        mm->add_instruction(migraphx::op::add{}, l1, l2);
        migraphx::quantize_fp16(p, {"all"});
        return p;
    };
};
