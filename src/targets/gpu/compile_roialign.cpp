#include <migraphx/gpu/compile_roialign.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const roialign_kernel = R"__migraphx__(
#include <migraphx/kernels/roialign.hpp>
#include <migraphx/kernels/basic_ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

__global__ void roialign_kernel(void* in_x, void* in_rois, void* in_ind, void* y) 
{
    make_tensors()(in_x, in_rois, in_ind, y)([](auto&&... xs) {
        auto settings = make_roalign_settings(MIGRAPHX_MAKE_CONSTANT(float{ROIS_OFFSET}),
                                              _c<bool{IS_AVG_POOLING}>,
                                              _c<int64_t{SAMPLING_RATIO}>, 
                                              MIGRAPHX_MAKE_CONSTANT(float{SPATIAL_SCALE}));
        roialign(xs..., settings); 
    });
}

}

} // namespace migraphx

int main() {}

)__migraphx__";

operation compile_roialign(context&, const std::vector<shape>& io_shapes, const value& val)
{
    hip_compile_options options;
    auto out_s             = io_shapes.back();
    options.local          = 128;
    options.global         = compute_global(out_s.elements(), options.local);
    options.inputs         = io_shapes;
    options.output         = out_s;
    options.kernel_name    = "roialign_kernel";
    options.virtual_inputs = io_shapes;

    // sampling_ratio
    assert(val.contains("sampling_ratio"));
    auto sampling_ratio = val.at("sampling_ratio").to<int64_t>();
    options.params += " -DSAMPLING_RATIO=" + std::to_string(sampling_ratio);

    // pooling_mode
    assert(val.contains("mode"));
    auto mode           = val.at("mode").to<std::string>();
    bool is_avg_pooling = (mode == "avg");
    options.params += " -DIS_AVG_POOLING=" + std::to_string(static_cast<int>(is_avg_pooling));

    // coord_trans_mode
    assert(val.contains("coordinate_transformation_mode"));
    auto ctm          = val.at("coordinate_transformation_mode").to<std::string>();
    float rois_offset = (ctm == "output_half_pixel") ? -0.5f : 0.0f;
    options.params += " -DROIS_OFFSET=" + std::to_string(rois_offset);

    // spatial_scale
    assert(val.contains("spatial_scale"));
    float spatial_scale = val.at("spatial_scale").to<float>();
    options.params += " -DSPATIAL_SCALE=" + std::to_string(spatial_scale);

    return compile_hip_code_object(roialign_kernel, options);
}
} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
