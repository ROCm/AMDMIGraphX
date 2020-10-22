
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_prelu_brcst : verify_program<test_prelu_brcst>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {6}};
        auto x   = p.add_parameter("x", s);
        auto slp = p.add_parameter("slp", s);
        auto r   = p.add_instruction(migraphx::op::prelu{}, x, slp);
        p.add_return({r});

        return p;
    }
};


