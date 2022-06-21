
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/common.hpp>

struct test_roialign_nondefault : verify_program<test_roialign_nondefault>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape x_s{migraphx::shape::float_type, {5, 4, 10, 10}};

        migraphx::shape roi_s{migraphx::shape::float_type, {5, 4}};

        migraphx::shape ind_s{migraphx::shape::int64_type, {5}};
        std::vector<int64_t> ind_vec = {0, 2, 3, 4, 1};

        auto x   = mm->add_parameter("x", x_s);
        auto roi = mm->add_parameter("roi", roi_s);
        auto ind = mm->add_literal(migraphx::literal(ind_s, ind_vec));
        auto r   = mm->add_instruction(
            migraphx::make_op("roialign",
                              {{"coordinate_transformation_mode", "output_half_pixel"},
                               {"mode", migraphx::op::pooling_mode::max},
                               {"spatial_scale", 1.0},
                               {"output_height", 5},
                               {"output_width", 5},
                               {"sampling_ratio", 2}}),
            x,
            roi,
            ind);
        mm->add_return({r});

        return p;
    }
};
