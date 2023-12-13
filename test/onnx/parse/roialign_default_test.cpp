
#include <onnx_test.hpp>


TEST_CASE(roialign_default_test)
{
    migraphx::shape sx{migraphx::shape::float_type, {10, 4, 7, 8}};
    migraphx::shape srois{migraphx::shape::float_type, {8, 4}};
    migraphx::shape sbi{migraphx::shape::int64_type, {8}};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto x    = mm->add_parameter("x", sx);
    auto rois = mm->add_parameter("rois", srois);
    auto bi   = mm->add_parameter("batch_ind", sbi);

    // Due to the onnx model using opset 12, the coordinate_transformation_mode should be set to
    // output_half_pixel
    auto r = mm->add_instruction(
        migraphx::make_op("roialign", {{"coordinate_transformation_mode", "output_half_pixel"}}),
        x,
        rois,
        bi);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("roialign_default_test.onnx");

    EXPECT(p == prog);
}


