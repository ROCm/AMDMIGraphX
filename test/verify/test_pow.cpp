
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_pow : verify_program<test_pow>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {6}};
        std::vector<float> vec_e(s.elements(), 2.0f);
        auto b = p.add_parameter("x", s);
        auto e = p.add_literal(migraphx::literal(s, vec_e));
        p.add_instruction(migraphx::op::pow{}, b, e);
        return p;
    }
};


