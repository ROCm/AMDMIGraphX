#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_scatternd_add : verify_program<test_scatternd_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {8}};
        migraphx::shape is{itype, {1, 4}};
        migraphx::shape us{dtype, {4}};
        std::vector<int64_t> ind_vec{4, 3, 1, 7};

        auto data    = mm->add_parameter("data", ds);
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto t_ind =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), indices);
        auto updates = mm->add_parameter("update", us);
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_add"), data, t_ind, updates);
        mm->add_return({scatternd});

        return p;
    }
};
