
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_topk_2 : verify_program<test_topk_2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 5}};
        auto data = mm->add_parameter("data", s);
        auto r    = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", 1}, {"k", 4}, {"largest", 0}}), data);
        auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        mm->add_return({r0});

        return p;
    }
};
