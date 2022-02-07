
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_gathernd : verify_program<test_gathernd>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape ds{migraphx::shape::float_type, {2, 3, 2, 3}};
        migraphx::shape is{migraphx::shape::int64_type, {2, 3, 2}};
        std::vector<int> indices{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
        auto a0        = mm->add_parameter("data", ds);
        auto a1        = mm->add_literal(migraphx::literal{is, indices});
        int batch_dims = 1;
        mm->add_instruction(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), a0, a1);
        return p;
    }
};
