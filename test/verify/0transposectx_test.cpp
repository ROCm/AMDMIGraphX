
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_transposectx : verify_program<test_transposectx>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {16, 12, 384, 64}});
        mm->add_instruction(migraphx::make_op("transposectx"), x);
        p.debug_print();
        return p;
    }
};
