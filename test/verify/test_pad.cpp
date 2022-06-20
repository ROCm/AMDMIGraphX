
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_pad : verify_program<test_pad>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s0{migraphx::shape::int32_type, {1, 96, 165, 165}};
        std::vector<int64_t> pads0 = {0, 0, 0, 0, 0, 0, 1, 1};
        std::vector<int64_t> pads1 = {0, 0, 0, 0, 1, 1, 1, 1};
        std::vector<int64_t> pads2 = {1, 1, 1, 1, 0, 0, 0, 0};
        std::vector<int64_t> pads3 = {1, 0, 1, 0, 1, 0, 2, 0};
        auto l0                    = mm->add_parameter("x", s0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads0}}), l0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads1}}), l0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads2}}), l0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads3}}), l0);
        return p;
    }
};
