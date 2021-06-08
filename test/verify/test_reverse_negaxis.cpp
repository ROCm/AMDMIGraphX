#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_reverse_negaxis : verify_program<test_reverse_negaxis>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {4, 16}};
        auto a0                   = mm->add_parameter("data", s);
        std::vector<int64_t> axis = {-1};
        mm->add_instruction(migraphx::make_op("reverse", {{"axes", axis}}), a0);
        return p;
    }
};
