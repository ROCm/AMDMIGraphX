
#include <onnx_test.hpp>

TEST_CASE(nms_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::float_type, {1, 6, 4}};
    auto b = mm->add_parameter("boxes", sb);

    migraphx::shape ss{migraphx::shape::float_type, {1, 1, 6}};
    auto s = mm->add_parameter("scores", ss);

    migraphx::shape smo{migraphx::shape::int64_type, {1}};
    auto mo = mm->add_parameter("max_output_boxes_per_class", smo);

    migraphx::shape siou{migraphx::shape::float_type, {1}};
    auto iou = mm->add_parameter("iou_threshold", siou);

    migraphx::shape sst{migraphx::shape::float_type, {1}};
    auto st = mm->add_parameter("score_threshold", sst);

    auto ret = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}), b, s, mo, iou, st);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("nms_test.onnx");
    EXPECT(p == prog);
}
