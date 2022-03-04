#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_prefix_scan_sum_2d_small : verify_program<test_prefix_scan_sum_2d_small>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1}};
        auto x = mm->add_parameter("x", s);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), x);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), xb);
        return p;
    }
};

struct test_prefix_scan_sum_2d_large : verify_program<test_prefix_scan_sum_2d_large>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 1000}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), x);
        return p;
    }
};
