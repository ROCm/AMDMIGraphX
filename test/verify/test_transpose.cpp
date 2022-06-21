
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_transpose : verify_program<test_transpose>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {4, 3, 4, 4}};
        auto x                    = mm->add_parameter("x", s);
        std::vector<int64_t> perm = {0, 2, 3, 1};
        auto l = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), x);
        mm->add_instruction(migraphx::make_op("contiguous"), l);
        return p;
    }
};
