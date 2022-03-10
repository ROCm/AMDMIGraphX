#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/cpp_generator.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>

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

)__migraphx__";

struct roialign_compiler : compiler<roialign_compiler>
{
    std::vector<std::string> names() const { return {"roialign"}; }

    operation compile_op(context&, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.output         = inputs.back();
        options.local          = v.get("local", 128);;
        options.global         = v.get("global", compute_global(options.output.elements(), options.local));
        options.inputs         = inputs;
        options.kernel_name    = "roialign_kernel";

        // sampling_ratio
        options.params += " -DSAMPLING_RATIO=" + v.at("sampling_ratio").to<std::string>();
        
        // pooling_mode
        auto mode           = v.at("mode").to<migraphx::op::pooling_mode>();
        int is_avg_pooling = (mode == migraphx::op::pooling_mode::average);
        options.params += " -DIS_AVG_POOLING=" + std::to_string(is_avg_pooling);

        // coord_trans_mode
        auto ctm          = v.at("coordinate_transformation_mode").to<std::string>();
        float rois_offset = (ctm == "output_half_pixel") ? -0.5f : 0.0f;
        options.params += " -DROIS_OFFSET=" + std::to_string(rois_offset);
        
        // spatial_scale
        options.params += " -DSPATIAL_SCALE=" + v.at("spatial_scale").to<std::string>();

        return compile_hip_code_object(roialign_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, operation op) const
    {
        return replace(
            compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }

};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
