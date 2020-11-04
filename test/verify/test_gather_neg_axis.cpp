
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_gather_neg_axis : verify_program<test_gather_neg_axis>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {2, 2}};
        std::vector<int> indices{1, 2, 2, 1};
        auto a0  = p.add_parameter("data", s);
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        return p;
    }
};
