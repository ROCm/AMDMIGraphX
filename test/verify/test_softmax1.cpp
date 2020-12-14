
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_softmax1 : verify_program<test_softmax1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {5, 3, 3, 4}});
        mm->add_instruction(migraphx::make_op("softmax", {{"axis", 0}}), x);
        return p;
    }
};
