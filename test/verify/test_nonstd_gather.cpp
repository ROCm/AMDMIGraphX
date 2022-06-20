
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_nonstd_gather : verify_program<test_nonstd_gather>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {2, 2}};
        std::vector<int> indices{1, 1, 0, 2};
        auto d  = mm->add_parameter("data", s);
        auto td = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), d);
        auto ind = mm->add_literal(migraphx::literal{s_indices, indices});
        auto tind =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), ind);
        auto r = mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), td, tind);
        mm->add_return({r});

        return p;
    }
};
