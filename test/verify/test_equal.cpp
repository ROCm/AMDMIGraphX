
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_equal : verify_program<test_equal>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::double_type, {2, 3, 4, 6}};
        auto input1 = p.add_parameter("x", s);
        auto input2 = p.add_parameter("y", s);
        auto r      = p.add_instruction(migraphx::op::equal{}, input1, input2);
        p.add_return({r});
        return p;
    };
};


