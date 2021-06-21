
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_slice_reverse : verify_program<test_slice_reverse>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {3, 5}};
        auto x = mm->add_parameter("x", s);
        mm->add_literal({{migraphx::shape::int32_type, {2}}, {-1, 1}});
        auto slice_out = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 2}}, {"ends", {2, -1}}}),
            x);
        mm->add_instruction(migraphx::make_op("reverse", {{"axes", {0}}}), slice_out);

        return p;
    }
};
