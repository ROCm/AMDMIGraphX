
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_atanh : verify_program<test_atanh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::double_type, {16}};
        auto x       = mm->add_parameter("x", s);
        auto min_val = mm->add_literal(-0.95);
        auto max_val = mm->add_literal(0.95);
        min_val      = mm->add_instruction(migraphx::op::multibroadcast{{16}}, min_val);
        max_val      = mm->add_instruction(migraphx::op::multibroadcast{{16}}, max_val);
        auto cx      = mm->add_instruction(migraphx::op::clip{}, x, min_val, max_val);
        mm->add_instruction(migraphx::op::atanh{}, cx);
        return p;
    }
};
