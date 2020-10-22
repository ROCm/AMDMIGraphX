
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>
#include <cassert>

struct test_contiguous_broadcast_transpose : verify_program<test_contiguous_broadcast_transpose>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {1, 3072, 768}, {0, 1, 3072}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::contiguous{}, x);
        assert(p.get_output_shapes().back().standard());
        return p;
    }
};
