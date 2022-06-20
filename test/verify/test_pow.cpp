
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_pow : verify_program<test_pow>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {6}};
        std::vector<float> vec_e(s.elements(), 2.0f);
        auto b = mm->add_parameter("x", s);
        auto e = mm->add_literal(migraphx::literal(s, vec_e));
        mm->add_instruction(migraphx::make_op("pow"), b, e);
        return p;
    }
};
