
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_softmax2 : verify_program<test_softmax2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1000, 1, 1}});
        p.add_instruction(migraphx::op::softmax{}, x);
        return p;
    }
};
