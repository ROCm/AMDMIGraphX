
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_softmax3 : verify_program<test_softmax3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {5, 3, 3, 4}});
        auto sx = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 3}}, {"starts", {1, 1}}, {"ends", {5, 4}}}),
            x);
        auto r = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 0}}), sx);
        mm->add_return({r});
        return p;
    }
};
