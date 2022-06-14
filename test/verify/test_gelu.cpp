
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_gelu : verify_program<test_gelu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<size_t> input_lens{1, 1, 5};
        auto x           = mm->add_parameter("x", {migraphx::shape::float_type, input_lens});
        auto half        = mm->add_literal(0.5f);
        auto one         = mm->add_literal(1.0f);
        auto sqrt2       = mm->add_literal(static_cast<float>(M_SQRT2));
        auto half_mbcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), half);
        auto mul_half     = mm->add_instruction(migraphx::make_op("mul"), x, half_mbcast);
        auto sqrt2_mbcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), sqrt2);
        auto div        = mm->add_instruction(migraphx::make_op("div"), x, sqrt2_mbcast);
        auto erf        = mm->add_instruction(migraphx::make_op("erf"), div);
        auto one_mbcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), one);
        auto add_one = mm->add_instruction(migraphx::make_op("add"), erf, one_mbcast);
        mm->add_instruction(migraphx::make_op("mul"), mul_half, add_one);
        return p;
    }
};
