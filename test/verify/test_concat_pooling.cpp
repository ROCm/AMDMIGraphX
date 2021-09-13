
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_concat_pooling : verify_program<test_concat_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 256, 8, 8}});
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), input);
        auto concat   = mm->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), transpose);
        auto concat_t = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), concat);

        auto pooling = mm->add_instruction(migraphx::make_op("pooling",
                                                             {{"mode", "average"},
                                                              {"padding", {0, 0}},
                                                              {"stride", {1, 1}},
                                                              {"lengths", {8, 8}}}),
                                           concat_t);
        mm->add_instruction(migraphx::make_op("relu"), pooling);
        return p;
    }
};
