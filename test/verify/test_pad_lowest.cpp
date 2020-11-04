
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_pad_lowest : verify_program<test_pad_lowest>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<migraphx::half> data0(4);
        std::iota(data0.begin(), data0.end(), 0);
        migraphx::shape s0{migraphx::shape::half_type, {2, 2}};
        auto l0 = p.add_literal(migraphx::literal{s0, data0});
        migraphx::op::pad op{};
        op.value = std::numeric_limits<float>::lowest();
        op.pads  = {0, 0, 1, 1};
        p.add_instruction(op, l0);
        return p;
    }
};
