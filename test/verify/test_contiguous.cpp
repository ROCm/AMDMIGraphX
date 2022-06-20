
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

#include <cassert>

struct test_contiguous : verify_program<test_contiguous>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {4, 4, 4, 3}, {48, 4, 1, 16}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(migraphx::make_op("contiguous"), x);
        assert(p.get_output_shapes().back().standard());
        return p;
    }
};
