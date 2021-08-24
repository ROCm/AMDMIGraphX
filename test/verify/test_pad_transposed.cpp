
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_pad_transposed : verify_program<test_pad_transposed>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {1, 224, 224, 3}};
        auto x = mm->add_parameter("x", s);
        auto t =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), x);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 2, 2, 0, 0, 3, 3}}}), t);
        return p;
    }
};
