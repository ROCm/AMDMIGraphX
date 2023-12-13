
#include <onnx_test.hpp>

TEST_CASE(roialign_test)
{
    migraphx::shape sx{migraphx::shape::float_type, {10, 5, 4, 7}};
    migraphx::shape srois{migraphx::shape::float_type, {8, 4}};
    migraphx::shape sbi{migraphx::shape::int64_type, {8}};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto x    = mm->add_parameter("x", sx);
    auto rois = mm->add_parameter("rois", srois);
    auto bi   = mm->add_parameter("batch_ind", sbi);

    auto r = mm->add_instruction(
        migraphx::make_op("roialign",
                          {{"coordinate_transformation_mode", "output_half_pixel"},
                           {"spatial_scale", 2.0f},
                           {"output_height", 5},
                           {"output_width", 5},
                           {"sampling_ratio", 3}}),
        x,
        rois,
        bi);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("roialign_test.onnx");

    EXPECT(p == prog);
}
