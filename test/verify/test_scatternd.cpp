#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_scatternd : verify_program<test_scatternd>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {1}};
        migraphx::shape is{itype, {4, 1}};
        migraphx::shape us{dtype, {4}};
        std::vector<int64_t> ind_vec{4, 3, 1, 7};

        auto ld = mm->add_literal(migraphx::literal{ds, {1}});
        auto data =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}), ld);
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_parameter("update", us);
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
        mm->add_return({scatternd});

        return p;
    }
};
