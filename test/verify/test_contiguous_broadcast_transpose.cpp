
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

#include <cassert>

struct test_contiguous_broadcast_transpose : verify_program<test_contiguous_broadcast_transpose>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 3072, 768}, {0, 1, 3072}};
        auto x = mm->add_parameter("x", s);
        mm->add_instruction(migraphx::make_op("contiguous"), x);
        assert(p.get_output_shapes().back().standard());
        return p;
    }
};
