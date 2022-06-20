
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_rsqrt : verify_program<test_rsqrt>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<size_t> input_lens{1, 3, 16, 16};
        migraphx::shape s{migraphx::shape::float_type, input_lens};
        auto x       = mm->add_parameter("x", s);
        auto min_val = mm->add_literal(1.0f);
        auto max_val = mm->add_literal(std::numeric_limits<float>::max());
        min_val      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), min_val);
        max_val = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), max_val);
        auto l0 = mm->add_instruction(migraphx::make_op("clip"), x, min_val, max_val);
        mm->add_instruction(migraphx::make_op("rsqrt"), l0);
        return p;
    };
};
